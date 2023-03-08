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

  void enqueue(InnerTask *task) {
    this->mixed_queue.push_back(task);
    num_tasks++;
  };

  /*
  Returns the next task that is ready to dequeue on this device.
  If there are no tasks that can dequeued, returns nullptr.

  A task is ready to dequeue if:
  1. It is a single-device task
  2. It is a multi-device task that is no longer waiting for its other
  instances

  We do not block on multi-device tasks that are still waiting for their other
  instances.

  This does not check resources, it only checks if the task is ready to
  dequeue. It does not remove the returned task from the queue.
  */
  InnerTask *front() {
    // First, check any waiting multi-device tasks
    if (!waiting_queue.empty()) {
      InnerTask *head = waiting_queue.front();
      int waiting_count = head->get_num_instances<category>();

      // Any MD task that is no longer waiting should be blocking
      if (waiting_count < 1) {
        // Remove from waiting queue if dequeued by last instance
        if (head->get_removed<category>()) {
          // TODO(wlr): Should I remove this here?
          waiting_queue.pop_front();

          // TODO(wlr): Should num_tasks include waiting tasks?
          // this->num_tasks--;
        }
        return nullptr;
      }
    }

    if (!mixed_queue.empty()) {
      InnerTask *head = mixed_queue.front();
      int prev_waiting_count = head->decrement_num_instances<category>();

      // Check if the task is waiting for other instances
      if (prev_waiting_count == 1) {
        return head;
      } else {
        // If the task is still waiting, move it to the waiting queue
        waiting_queue.push_back(head);
        mixed_queue.pop_front();

        // TODO(wlr): Should num_tasks include waiting tasks?
        this->num_tasks--;
      }
    }

    return nullptr;
  }

  InnerTask *pop() {
    InnerTask *task = front();
    if (task != nullptr) {
      mixed_queue.pop_front();
      task->set_removed<category>(true);
      num_tasks--;
    }
    return task;
  }

  inline size_t size() { return num_tasks.load(); }
  inline bool empty() { return mixed_queue.empty() && waiting_queue.empty(); }

protected:
  Device *device;
  MixedQueue_t mixed_queue;
  MDQueue_t waiting_queue;
  std::atomic<int> num_tasks = 0;
};

// TODO(wlr): I don't know what to name this.
template <ResourceCategory category> class PhaseManager {
public:
  PhaseManager() = default;

  PhaseManager(DeviceManager *devices) {
    for (const DeviceType dev_type : architecture_types) {
      int num_devices = devices->get_num_devices(dev_type);
      this->ndevices += num_devices;

      for (Device *device : devices->get_devices(dev_type)) {
        this->device_queues.push_back(DeviceQueue<category>(device));
      }
    }
  }

  void enqueue(InnerTask *task) {
    for (auto device : task->assigned_devices) {
      device_queues[device->get_global_id()].enqueue(task);
    }
    this->num_tasks++;
  }

  InnerTask *front() {
    // TODO(wlr): Hochan, can you check this?
    // I'm not sure if this is the right way to loop over dequeable tasks
    // Should we drain each device first, or try each device in
    // turn?

    int start_idx = last_device_idx;
    int end_idx = start_idx + ndevices;
    int current_idx = start_idx;

    bool has_task = this->size() > 0;
    while (has_task) {

      // Loop over all devices starting from after last success location
      for (int i = start_idx; i < end_idx; ++i) {
        current_idx = i % ndevices;

        // Try to get a non-waiting task
        InnerTask *task = device_queues[current_idx].front();
        if (task != nullptr) {
          last_device_idx = ++current_idx;
          return task;
        }
      }

      has_task = this->size() > 0;
    }

    // If we get here, there are no tasks that can be dequeued
    // This should only happen if called on an empty phase
    return nullptr;
  }

  InnerTask *pop() {
    InnerTask *task = device_queues[last_device_idx].pop();
    this->num_tasks--;
    return task;
  }

  inline size_t size() { return this->num_tasks.load(); }

protected:
  // TODO(wlr): I keep changing this back and forth.
  //  For now I think global indexing is easier to work with.
  // std::array<std::vector<DeviceQueue<category>>, NUM_DEVICE_TYPES>
  //     device_queues;
  std::vector<DeviceQueue<category>> device_queues;

  int last_device_idx = 0;
  DeviceType last_device_type = CPU;

  int ndevices = 0;
  std::atomic<int> num_tasks = 0;
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

  bool check_resources(InnerTask *task);
  void reserve_resources(InnerTask *task);
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

  bool check_resources(InnerTask *task);
  void reserve_resources(InnerTask *task);
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
