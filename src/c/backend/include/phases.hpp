#pragma once
#ifndef PARLA_PHASES_HPP
#define PARLA_PHASES_HPP

#include "containers.hpp"
#include "device.hpp"
#include "device_manager.hpp"
#include "device_queues.hpp"
#include "policy.hpp"
#include "resources.hpp"
#include "runtime.hpp"

#include <memory>
#include <string>

enum class MapperState { Failure = 0, Success = 1, MAX = 2 };
enum class MemoryReserverState { Failure = 0, Success = 1, MAX = 2 };
enum class RuntimeReserverState {
  Failure = 0,
  NoTask = 1,
  NoResource = 2,
  NoWorker = 3,
  Success = 4,
  MAX = 5
};
enum class LauncherState { Failure = 0, Success = 1, MAX = 2 };

template <typename S> class PhaseStatus {
protected:
  const int size{static_cast<int>(S::MAX)};
  std::string name{"Status"};

public:
  int status[static_cast<int>(S::MAX)];

  PhaseStatus() = default;
  PhaseStatus(std::string name) : name(name) {}

  void reset() {
    for (int i = 0; i < size; ++i) {
      this->status[i] = 0;
    }
  }

  inline void set(S state, int value) {
    this->status[static_cast<int>(state)] = value;
  }
  inline const int get(S state) const {
    return this->status[static_cast<int>(state)];
  }
  inline void increase(S state) { this->status[static_cast<int>(state)]++; }
  inline void decrease(S state) { this->status[static_cast<int>(state)]--; }

  void print() const {
    std::cout << this->name + "(";
    for (int i = 0; i < size; ++i) {
      std::cout << this->status[i];
    }
    std::cout << ")\n";
  }
};

class MapperStatus : public PhaseStatus<MapperState> {};
class MemoryReserverStatus : public PhaseStatus<MemoryReserverState> {};
class RuntimeReserverStatus : public PhaseStatus<RuntimeReserverState> {};
class LauncherStatus : public PhaseStatus<LauncherState> {};

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
  inline static const std::string name{"Phase"};
  std::mutex mtx;
  InnerScheduler *scheduler;
  DeviceManager *device_manager;
  TaskStateList enqueue_buffer;
};

/**
 * @brief Mapper phase of the scheduler. Uses constraints to assign tasks to
 * device sets.
 */
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
  inline static const std::string name{"Mapper"};
  MapperStatus status{name};
  TaskQueue mappable_tasks;
  std::vector<InnerTask *> mapped_tasks_buffer;
  uint64_t dummy_dev_idx_;

  std::shared_ptr<MappingPolicy> policy_;
};

/**
 * @brief MemoryReserver phase of the scheduler. Reserves all 'persistent
 * resources`. This plans task execution on the device set. Here all 'persistent
 * resources` that have a lifetime greater than the task body are reserved and
 * shared between tasks. At the moment this is only the memory a task uses. The
 * memory is reserved to allow input data to be prefetched onto the devices.
 */
class MemoryReserver : virtual public SchedulerPhase {
public:
  MemoryReserver(InnerScheduler *scheduler, DeviceManager *devices)
      : SchedulerPhase(scheduler, devices) {
    // std::cout << "MemoryReserver created\n";
    this->reservable_tasks =
        new PhaseManager<ResourceCategory::Persistent>(devices);
  }

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();

protected:
  // std::string name{"Memory Reserver"};
  PhaseManager<ResourceCategory::Persistent> *reservable_tasks;
  inline static const std::string name{"Memory Reserver"};
  MemoryReserverStatus status{name};
  std::vector<InnerTask *> reserved_tasks_buffer;

  bool check_resources(InnerTask *task);
  void reserve_resources(InnerTask *task);
};

/**
 * @brief RuntimeReserver phase of the scheduler. Reserves all 'non-persistent
 * resources`. This plans task execution on the device set. Here all
 * 'non-persistent resources` that have a lifetime equal to the task body are
 * reserved and are not directly shared between tasks. At the moment this is
 * only the VCUS/Threads a task uses.
 * This phase submits the task to the launcher.
 */
class RuntimeReserver : virtual public SchedulerPhase {
public:
  RuntimeReserver(InnerScheduler *scheduler, DeviceManager *devices)
      : SchedulerPhase(scheduler, devices) {
    // std::cout << "RuntimeReserver created" << std::endl;
    this->runnable_tasks =
        new PhaseManager<ResourceCategory::NonPersistent>(devices);
  }

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();
  PhaseManager<ResourceCategory::NonPersistent> *get_runnable_tasks() {
    return this->runnable_tasks;
  }

  const std::string &get_name() const { return this->name; }
  const RuntimeReserverStatus &get_status() const { return this->status; }
  const void print_status() const { this->status.print(); }

protected:
  PhaseManager<ResourceCategory::NonPersistent> *runnable_tasks;
  inline static const std::string name{"Runtime Reserver"};
  RuntimeReserverStatus status{name};
  std::vector<InnerTask *> launchable_tasks_buffer;

  bool check_resources(InnerTask *task);
  void reserve_resources(InnerTask *task);
};

#ifdef PARLA_ENABLE_LOGGING
LOG_ADAPT_STRUCT(RuntimeReserver, print_status)
#endif

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
  inline static const std::string name{"Launcher"};
  LauncherStatus status{name};
  /*Buffer to store not yet launched tasks. Currently unused. Placeholder in
   * case it becomes useful.*/
  TaskList task_buffer;
  WorkerList worker_buffer;
};

#endif // PARLA_PHASES_HPP
