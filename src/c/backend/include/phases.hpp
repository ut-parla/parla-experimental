#pragma once
#ifndef PARLA_PHASES_HPP
#define PARLA_PHASES_HPP

#include "atomic_wrapper.hpp"
#include "containers.hpp"
#include "device.hpp"
#include "device_manager.hpp"
#include "device_queues.hpp"
#include "policy.hpp"
#include "rl_task_mapper.hpp"
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

class RLEnvironment;

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
  Mapper() = delete;
  Mapper(InnerScheduler *scheduler, DeviceManager *devices,
         PArrayTracker *parray_tracker, MappingPolicyType policy_type);

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();

  /// Increase the number of the tasks mapped to a device.
  ///
  /// @param dev_id Device global ID where a task is mapped
  void atomic_incr_num_mapped_tasks_device(DevID_t dev_id, double  weight = 1) {
    // Increase the number of the total mapped tasks to the whole devices.
    // We do not get the old total number of mapped tasks.
    this->num_mapped_tasks_mtx_.lock();
    this->total_num_mapped_tasks_ += weight;
    (this->dev_num_mapped_tasks_)[dev_id] += weight;
    std::cout << "Incr counter Weight:" << weight << " is added:" <<
      this->total_num_mapped_tasks_ <<
      ", dev:" << dev_id << " to " << (this->dev_num_mapped_tasks_)[dev_id] << "\n";
    this->num_mapped_tasks_mtx_.unlock();
  }

  /// Decrease the number of the tasks mapped to a device.
  ///
  /// @param dev_id Device global ID where a task is mapped
  void atomic_decr_num_mapped_tasks_device(DevID_t dev_id, double weight = 1) {
    // Decrease the number of the total mapped tasks to the whole devices.
    // We do not get the old total number of mapped tasks.
    this->num_mapped_tasks_mtx_.lock();
    this->total_num_mapped_tasks_ -= weight;
    (this->dev_num_mapped_tasks_)[dev_id] -= weight;
    std::cout << "Decr counter Weight:" << weight << " is added:" <<
      this->total_num_mapped_tasks_ <<
      ", dev:" << dev_id << " to " << (this->dev_num_mapped_tasks_)[dev_id] << "\n";

    this->num_mapped_tasks_mtx_.unlock();
  }

  /// Return the number of total mapped tasks to the whole devices.
  ///
  /// @return The old number of total mapped tasks
  double atomic_load_total_num_mapped_tasks() {
    double total_num_mapped_tasks{0};
    this->num_mapped_tasks_mtx_.lock();
    total_num_mapped_tasks = this->total_num_mapped_tasks_; 
    this->num_mapped_tasks_mtx_.unlock();
    return total_num_mapped_tasks;
  }

  /// Return the number of mapped tasks to a single device.
  ///
  /// @param dev_id Device global ID where a task is mapped
  /// @return The old number of the tasks mapped to a device
  double atomic_load_dev_num_mapped_tasks_device(DevID_t dev_id) {
    double dev_num_mapped_tasks{0};
    this->num_mapped_tasks_mtx_.lock();
    dev_num_mapped_tasks = (this->dev_num_mapped_tasks_)[dev_id];
    this->num_mapped_tasks_mtx_.unlock();
    return dev_num_mapped_tasks;
  }

  /// @brief Return a raw pointer to a policy.
  /// @detail It exposes a mapping policy object to enable programmers to
  /// call policy-specific features.
  /// For example, in the RL policy, a launcher needs to call and add time
  /// inforamtion at the launching phase.
  MappingPolicy* get_policy_raw_pointer() {
    return this->policy_.get();
  }

protected:
  inline static const std::string name{"Mapper"};
  MapperStatus status{name};
  TaskQueue mappable_tasks;
  std::vector<InnerTask *> mapped_tasks_buffer;
  uint64_t dummy_dev_idx_;

  std::shared_ptr<MappingPolicy> policy_;
  /// The total loads of tasks mapped to and running on the whole devices.
  double total_num_mapped_tasks_{0};
  std::mutex num_mapped_tasks_mtx_;
  /// The total number of tasks mapped to and running on a single device.
  std::vector<double> dev_num_mapped_tasks_;
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
        std::make_shared<PhaseManager<ResourceCategory::Persistent>>(devices);
  }

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();

protected:
  // std::string name{"Memory Reserver"};
  std::shared_ptr<PhaseManager<ResourceCategory::Persistent>> reservable_tasks;
  inline static const std::string name{"Memory Reserver"};
  MemoryReserverStatus status{name};
  std::vector<InnerTask *> reserved_tasks_buffer;

  bool check_resources(InnerTask *task);
  void reserve_resources(InnerTask *task);
  void create_datamove_tasks(InnerTask *task);
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
    // FIXME: This leaks memory. Need to add deconstructor.
    this->runnable_tasks =
        std::make_shared<PhaseManager<ResourceCategory::NonPersistent>>(
            devices);
    this->movement_tasks =
        std::make_shared<PhaseManager<ResourceCategory::Movement>>(devices);
  }

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();
  size_t get_compute_count();
  size_t get_movement_count();

  const std::string &get_name() const { return this->name; }
  const RuntimeReserverStatus &get_status() const { return this->status; }
  const void print_status() const { this->status.print(); }

protected:
  std::shared_ptr<PhaseManager<ResourceCategory::NonPersistent>> runnable_tasks;
  std::shared_ptr<PhaseManager<ResourceCategory::Movement>> movement_tasks;

  inline static const std::string name{"Runtime Reserver"};
  RuntimeReserverStatus status{name};
  std::vector<InnerTask *> launchable_tasks_buffer;

  bool check_resources(InnerTask *task);
  bool check_data_resources(InnerTask *task);

  void reserve_resources(InnerTask *task);
  void reserve_data_resources(InnerTask *task);
};

class Launcher : virtual public SchedulerPhase {
public:
  /*Number of running tasks. A task is running if it has been assigned to a
   * worker and is not complete*/
  std::atomic<size_t> num_running_tasks{0};

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
