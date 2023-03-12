#pragma once
#include "resources.hpp"
#ifndef PARLA_PHASES_HPP
#define PARLA_PHASES_HPP

#include "containers.hpp"
#include "device.hpp"
#include "device_manager.hpp"
#include "device_queues.hpp"
#include "policy.hpp"
#include "runtime.hpp"
#include <memory>
#include <string>

class SchedulerPhase {
public:
  SchedulerPhase() = default;
  SchedulerPhase(InnerScheduler *scheduler, DeviceManager *devices)
      : scheduler(scheduler), device_manager(devices) {}

  virtual void enqueue(InnerTask *task) = 0;
  virtual void enqueue(std::vector<InnerTask *> &tasks) = 0;
  virtual void run(SchedulerPhase *next_phase) = 0;
  virtual size_t get_count() = 0;

protected:
  // std::string name{"Phase"};
  std::mutex mtx;
  InnerScheduler *scheduler;
  DeviceManager *device_manager;
  TaskStateList enqueue_buffer;
};

class Mapper : virtual public SchedulerPhase {
public:
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
  // std::string name{"Mapper"};
  TaskQueue mappable_tasks;
  std::vector<InnerTask *> mapped_tasks_buffer;
  uint64_t dummy_dev_idx_;

  std::shared_ptr<MappingPolicy> policy_;
};

class MemoryReserver : virtual public SchedulerPhase {
public:
  MemoryReserver(InnerScheduler *scheduler, DeviceManager *devices)
      : SchedulerPhase(scheduler, devices) {
    std::cout << "MemoryReserver created\n";
    this->reservable_tasks =
        new PhaseManager<ResourceCategory::PERSISTENT>(devices);
  }

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();

protected:
  // std::string name{"Memory Reserver"};
  PhaseManager<ResourceCategory::PERSISTENT> *reservable_tasks;
  std::vector<InnerTask *> reserved_tasks_buffer;

  bool check_resources(InnerTask *task);
  void reserve_resources(InnerTask *task);
};

class RuntimeReserver : virtual public SchedulerPhase {
public:
  RuntimeReserver(InnerScheduler *scheduler, DeviceManager *devices)
      : SchedulerPhase(scheduler, devices) {
    std::cout << "RuntimeReserver created" << std::endl;
    this->runnable_tasks =
        new PhaseManager<ResourceCategory::NON_PERSISTENT>(devices);
  }

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();
  PhaseManager<ResourceCategory::NON_PERSISTENT> *get_runnable_tasks() {
    return this->runnable_tasks;
  }

protected:
  // std::string name{"Runtime Reserver"};
  PhaseManager<ResourceCategory::NON_PERSISTENT> *runnable_tasks;
  std::vector<InnerTask *> launchable_tasks_buffer;

  bool check_resources(InnerTask *task);
  void reserve_resources(InnerTask *task);
};

class Launcher : virtual public SchedulerPhase {
public:
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
  // std::string name{"Launcher"};
  /*Buffer to store not yet launched tasks. Currently unused. Placeholder in
   * case it becomes useful.*/
  TaskList task_buffer;
  WorkerList worker_buffer;
};

#endif // PARLA_PHASES_HPP
