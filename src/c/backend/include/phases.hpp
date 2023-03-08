#pragma once
#include "resources.hpp"
#ifndef PARLA_PHASES_HPP
#define PARLA_PHASES_HPP

#include "containers.hpp"
#include "device.hpp"
#include "device_manager.hpp"
#include "policy.hpp"
#include "runtime.hpp"

#include <memory>
#include <string>

template <ResourceCategory category> class DeviceQueue {
  using MixedQueue_t = TaskQueue;
  using MDQueue_t = TaskQueue;

public:
  DeviceQueue() = default;
  DeviceQueue(Device *device) : device(device) {}

  void enqueue(InnerTask *task) { this->mixed_queue.push_back(task); };

  InnerTask *next() {
    // TODO(wlr): Is there a way to do this much more efficiently?
    // This is not good code, but I'm not sure how to fix it.

    // First, check any waiting multi-device tasks
    if (!md_queue.empty()) {
      InnerTask *md_head = md_queue.front();
      int waiting_count = md_head->get_num_instances<category>();

      // Any MD task that is no longer waiting should be blocking
      if (waiting_count < 1) {
        if (md_head->get_removed<category>()) {
          // if the task has already been launched by another device, remove it
          md_queue.front_and_pop();
        }
        return nullptr;
      }
    }

    if (!mixed_queue.empty()) {
      InnerTask *mixed_head = mixed_queue.front();
      // Decrease the waiting count
      int prev_waiting_count = mixed_head->decrement_num_instances<category>();

      // Check if the task is waiting
      if (prev_waiting_count == 1) {
        // If the task is no longer waiting, check if it is launchable
        bool launchable = check_launchable(mixed_head);
        if (launchable) {
          mixed_queue.pop_front();
          mixed_head->set_removed<category>(true);
          return mixed_head;
        }
      } else {
        // If the task is still waiting for its other instances,
        // add to the md_queue
        md_queue.push_back(mixed_head);
        mixed_queue.pop_front();
      }
    }

    return nullptr;
  }
  inline size_t size() { return mixed_queue.size() + md_queue.size(); }
  inline bool empty() { return mixed_queue.empty() && md_queue.empty(); }

protected:
  Device *device;
  MixedQueue_t mixed_queue;
  MDQueue_t md_queue;

  bool check_launchable(InnerTask *task) {
    bool launchable = true;
    ResourcePool_t device_pool = device->get_reserved_pool();

    for (auto &device : task->device_constraints) {
      ResourcePool_t task_pool = device.second;
      launchable &= device_pool.check_greater<category>(task_pool);
    }

    return launchable;
  }
};

template <ResourceCategory category> class PhaseManager {

protected:
  // std::array<std::vector<DeviceQueue<category>>, NUM_DEVICE_TYPES>
  //     device_queues;
  std::vector<DeviceQueue<category>> device_queues;
  int last_device_idx = 0;
  int ndevices = 0;
  std::atomic<int> size = 0;

public:
  PhaseManager() = default;
  PhaseManager(DeviceManager *devices) {
    for (auto device : devices->get_devices<DeviceType::ANY>()) {
      this->device_queues[device->get_global_id()] =
          DeviceQueue<category>(device);
    }
    this->ndevices = device_queues.size();
  }

  void enqueue(InnerTask *task) {
    for (auto device : task->assigned_devices) {
      device_queues[device->get_global_id()].enqueue(task);
    }
    this->size++;
  }

  InnerTask *next(Device *device) {
    // TODO(wlr): I have no idea how this should iterate over the queues
    // I'm just going to do a round-robin over the devices once for now.
    int start_idx = last_device_idx;
    int end_idx = last_device_idx + size;

    for (int i = start_idx; i < end_idx; ++i) {
      int idx = i % ndevices;
      InnerTask *task = device_queues[idx].next();
      if (task != nullptr) {
        last_device_idx = idx;
        this->size--;
        return task;
      }
    }

    return nullptr;
  }
};

class PhaseStatus {
protected:
  const static int size = 3;
  std::string name = "Status";

public:
  int status[size];

  void reset() {
    for (int i = 0; i < size; ++i) {
      this->status[i] = 0;
    }
  }

  void set(int index, int value) { this->status[index] = value; }
  int get(int index) { return this->status[index]; }
  void increase(int state) { this->status[state]++; }

  void print() {
    std::cout << this->name + "(";
    for (int i = 0; i < size; ++i) {
      std::cout << this->status[i];
    }
    std::cout << ")\n";
  }
};

class SchedulerPhase {
public:
  SchedulerPhase() = default;
  SchedulerPhase(InnerScheduler *scheduler, DeviceManager *devices)
      : scheduler(scheduler), device_manager(devices) {}

  virtual void enqueue(InnerTask *task) = 0;
  virtual void enqueue(std::vector<InnerTask *> &tasks) = 0;
  virtual void run(SchedulerPhase *next_phase) = 0;
  virtual size_t get_count() = 0;

  PhaseStatus status;

protected:
  std::string name = "Phase";
  std::mutex mtx;
  InnerScheduler *scheduler;
  DeviceManager *device_manager;
  TaskStateList enqueue_buffer;
};

namespace Map {

enum State { failure, success };
class Status : public PhaseStatus {
protected:
  const static int size = 2;
  std::string name = "Mapper";
};

} // namespace Map

class Mapper : virtual public SchedulerPhase {
public:
  Map::Status status;
  Mapper() : SchedulerPhase(), dummy_dev_idx_{0} {}

  Mapper(InnerScheduler *scheduler, DeviceManager *devices,
         std::shared_ptr<MappingPolicy> policy)
      : SchedulerPhase(scheduler, devices), dummy_dev_idx_{0}, policy_{policy} {
  }

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();

protected:
  std::string name = "Mapper";
  TaskQueue mappable_tasks;
  std::vector<InnerTask *> mapped_tasks_buffer;
  uint64_t dummy_dev_idx_;

  std::shared_ptr<MappingPolicy> policy_;
};

namespace Reserved {

enum State { failure, success };

class Status : public PhaseStatus {
protected:
  const static int size = 2;
  std::string name = "MemoryReserver";
};
} // namespace Reserved

class MemoryReserver : virtual public SchedulerPhase {
public:
  Reserved::Status status;

  MemoryReserver(InnerScheduler *scheduler, DeviceManager *devices)
      : SchedulerPhase(scheduler, devices) {}

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();

protected:
  std::string name = "Memory Reserver";
  PhaseManager<ResourceCategory::PERSISTENT> reservable_tasks;
  std::vector<InnerTask *> reserved_tasks_buffer;
};

namespace Ready {

enum State { entered, task_miss, resource_miss, worker_miss, success };

class Status : virtual public PhaseStatus {
protected:
  const static int size = 5;
  std::string name = "RuntimeReserver";
};
} // namespace Ready

#ifdef PARLA_ENABLE_LOGGING
LOG_ADAPT_STRUCT(Ready::Status, status)
#endif

class RuntimeReserver : virtual public SchedulerPhase {

public:
  Ready::Status status;

  RuntimeReserver(InnerScheduler *scheduler, DeviceManager *devices)
      : SchedulerPhase(scheduler, devices) {}

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();

protected:
  std::string name = "Runtime Reserver";
  PhaseManager<ResourceCategory::NON_PERSISTENT> runnable_tasks;
  std::vector<InnerTask *> launchable_tasks_buffer;
};

#ifdef PARLA_ENABLE_LOGGING
LOG_ADAPT_STRUCT(RuntimeReserver, status)
#endif

namespace Launch {

enum State { failure, success };

class Status : public PhaseStatus {
protected:
  const static int size = 2;
  std::string name = "Launcher";
};
} // namespace Launch

class Launcher : virtual public SchedulerPhase {
public:
  Launch::Status status;

  /*Number of running tasks. A task is running if it has been assigned to a
   * worker and is not complete*/
  std::atomic<size_t> num_running_tasks;

  Launcher(InnerScheduler *scheduler, DeviceManager *devices)
      : SchedulerPhase(scheduler, devices) {}

  /*Add a task to the launcher. Currently this acquires the GIL and dispatches
   * the work to a Python Worker for each task */
  void enqueue(InnerTask *task){};
  void enqueue(InnerTask *task, InnerWorker *worker);
  void enqueue(std::vector<InnerTask *> &tasks){};

  /* A placeholder function in case work needs to be done at this stage. For
   * example, dispatching a whole buffer of tasks*/
  void run();
  void run(SchedulerPhase *next_phase) { this->run(); };

  /* Number of running tasks. A task is running if it has been assigned to a
   * worker and is not complete */
  size_t get_count() { return this->num_running_tasks.load(); }

protected:
  std::string name = "Launcher";
  /*Buffer to store not yet launched tasks. Currently unused. Placeholder in
   * case it becomes useful.*/
  TaskList task_buffer;
  WorkerList worker_buffer;
};

#endif // PARLA_PHASES_HPP
