/*! @file phases.hpp
 *  @brief Interface for scheduler runtime phases.
 *
 * This file contains the interface for scheduler runtime phases. This includes
 * classes for Task Mapping, Task Resource Reservation, and Task Launching.
 */

#pragma once
#ifndef PARLA_PHASES_HPP
#define PARLA_PHASES_HPP

#include "atomic_wrapper.hpp"
#include "containers.hpp"
#include "device.hpp"
#include "device_manager.hpp"
#include "device_queues.hpp"
#include "policy.hpp"
#include "resources.hpp"
#include "runtime.hpp"

#include <memory>
#include <string>

/*!
 * @brief Enum class for Task Mapping phase
 */
enum class MapperState { Failure = 0, Success = 1, MAX = 2 };

/*!
 * @brief Enum class for Task Resource Reservation phase (Persistent Resources)
 */
enum class MemoryReserverState { Failure = 0, Success = 1, MAX = 2 };

/*!
 * @brief Enum class for Task Resource Reservation phase (Non-Persistent
 * Resources)
 *
 * @details Counts the reasons for failure in the RuntimeReserver phase, such as
 * no available resource or no available worker.
 */
enum class RuntimeReserverState {
  Failure = 0,
  NoTask = 1,
  NoResource = 2,
  NoWorker = 3,
  Success = 4,
  MAX = 5
};
enum class LauncherState { Failure = 0, Success = 1, MAX = 2 };

/*!
 * @brief Records metrics that track phase execution (e.g. success, failure,
 * etc.)
 * @tparam S The enum class that defines the states to track.
 * @details Used to record counts of phase execution for tracing and debugging.
 * For example, the number of successful mappings per call.
 */
template <typename S> class PhaseStatus {
protected:
  const int size{static_cast<int>(S::MAX)};
  std::string name{"Status"};

public:
  int status[static_cast<int>(S::MAX)];

  PhaseStatus() = default;
  PhaseStatus(std::string name) : name(name) {}

  void reset() {
    /*!
     * @brief Resets the status array to 0.
     */
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

/*!
 * @brief Abstract Interface for general scheduler runtime phase.
 */
class SchedulerPhase {
public:
  SchedulerPhase() = default;

  /*!
   * @brief Constructor for the scheduler phase.
   * @param scheduler The scheduler that the phase belongs to.
   * @param devices The device manager that the phase uses.
   */
  SchedulerPhase(InnerScheduler *scheduler, DeviceManager *devices)
      : scheduler(scheduler), device_manager(devices) {}

  /*!
   * @brief Enqueue a task to the phase.
   */
  virtual void enqueue(InnerTask *task) = 0;

  /*!
   * @brief Enqueue a vector of tasks to the phase.
   */
  virtual void enqueue(std::vector<InnerTask *> &tasks) = 0;

  /*!
   * @brief Run the phase. Check tasks in the enqueued buffer, check phase
   * condition, and move tasks to the next phase.
   * @param next_phase The next phase to move tasks to.
   */
  virtual void run(SchedulerPhase *next_phase) = 0;

  /*!
   * @brief Get the number of tasks enqueued (and waiting) in the phase.
   */
  virtual size_t get_count() = 0;

protected:
  /// The name of the phase. Used for debugging and tracing.
  inline static const std::string name{"Phase"};
  /// Mutex lock for the phase (In case of a workerthread driven scheduler,
  /// ensure only 1 thread is running the phase at a time)
  std::mutex mtx;
  /// The scheduler that the phase belongs to.
  InnerScheduler *scheduler;
  /// The device manager that the phase uses.
  DeviceManager *device_manager;
  /// The number of tasks enqueued (and waiting) in the phase.
  TaskStatusList enqueue_buffer;
};

/**
 * @brief Mapper phase of the scheduler. Uses constraints to assign tasks to
 * device sets.
 */
class Mapper : virtual public SchedulerPhase {
public:
  Mapper() = delete;
  Mapper(InnerScheduler *scheduler, DeviceManager *devices,
         std::shared_ptr<MappingPolicy> policy)
      : SchedulerPhase(scheduler, devices), policy_{policy} {
    dev_num_mapped_tasks_.resize(devices->get_num_devices());
  }

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();


  /// @brief Increase the count of tasks mapped to a device
  /// @param dev_id Device global ID where a task is mapped
  /// @return The number of the tasks mapped to a device
  size_t atomic_incr_num_mapped_tasks_device(DevID_t dev_id,
                                             size_t weight = 1) {
    // Increase the number of the total mapped tasks to the whole devices.
    // We do not get the old total number of mapped tasks.
    total_num_mapped_tasks_.fetch_add(weight, std::memory_order_relaxed);
    return dev_num_mapped_tasks_[dev_id].fetch_add(weight,
                                                   std::memory_order_relaxed);
  }

  /// @brief Decrease the count of tasks mapped to a device.
  ///
  /// @param dev_id Device global ID where a task is mapped
  /// @return The number of the tasks mapped to a device
  size_t atomic_decr_num_mapped_tasks_device(DevID_t dev_id,
                                             size_t weight = 1) {
    // Decrease the number of the total mapped tasks to the whole devices.
    // We do not get the old total number of mapped tasks.
    total_num_mapped_tasks_.fetch_sub(weight, std::memory_order_relaxed);
    return dev_num_mapped_tasks_[dev_id].fetch_sub(weight,
                                                   std::memory_order_relaxed);
  }

  /// @brief Return the number of total mapped tasks to the whole devices.
  ///
  /// @return The old number of total mapped tasks
  const size_t atomic_load_total_num_mapped_tasks() const {
    return total_num_mapped_tasks_.load(std::memory_order_relaxed);
  }

  /// @brief Return the number of mapped tasks to a single device.
  ///
  /// @param dev_id Device global ID where a task is mapped
  /// @return The old number of the tasks mapped to a device
  const size_t atomic_load_dev_num_mapped_tasks_device(DevID_t dev_id) const {
    return dev_num_mapped_tasks_[dev_id].load(std::memory_order_relaxed);
  }

protected:
  /// The name of the phase. Used for debugging and tracing.
  inline static const std::string name{"Mapper"};
  /// The status of the phase. Used for debugging and tracing.
  MapperStatus status{name};
  /// The buffer of tasks enqueued (and waiting) in the phase.
  TaskQueue mappable_tasks;
  /// The buffer of tasks mapped to a device set. Waiting to be processed to add
  /// to the next phase
  std::vector<InnerTask *> mapped_tasks_buffer;
  /// The mapping policy object
  std::shared_ptr<MappingPolicy> policy_;
  /// The total number of tasks mapped to and running on the whole devices.
  std::atomic<size_t> total_num_mapped_tasks_{0};
  /// The total number of tasks mapped to and running on a single device.
  std::vector<CopyableAtomic<size_t>> dev_num_mapped_tasks_;
};

/**
 * @brief MemoryReserver phase of the scheduler. Reserves all 'persistent
 * resources`.
 *
 * @details This phase plans task execution on the device set. Here all
 * 'persistent resources` that have a lifetime greater than the task body are
 * reserved and shared between tasks. Typically this means the memory that the
 * task uses for both its input, output, and intermediate workspace. Memory
 * usage is planned and reserved ahead of task execution to allow input data to
 * be prefetched onto the devices through data movement tasks.
 *
 * @note Assumption: This is the only part of the runtime that decreases the
 * PersistentResources. Otherwise there will be race conditions without a lock
 * on the ResourcePool.
 */
class MemoryReserver : virtual public SchedulerPhase {
public:
  MemoryReserver(InnerScheduler *scheduler, DeviceManager *devices)
      : SchedulerPhase(scheduler, devices) {
    this->reservable_tasks =
        std::make_shared<PhaseManager<ResourceCategory::Persistent>>(devices);
  }

  void enqueue(InnerTask *task);
  void enqueue(std::vector<InnerTask *> &tasks);
  void run(SchedulerPhase *next_phase);
  size_t get_count();

protected:
  std::shared_ptr<PhaseManager<ResourceCategory::Persistent>> reservable_tasks;
  inline static const std::string name{"Memory Reserver"};
  MemoryReserverStatus status{name};
  std::vector<InnerTask *> reserved_tasks_buffer;

  /*!
   * @brief Check if the PersistentResources are available for a task.
   * @param task The task to check the PersistentResources for.
   */
  bool check_resources(InnerTask *task);

  /*!
   * @brief Reserve (decrease) the PersistentResources for a task.
   * @param task The task to reserve the PersistentResources for.
   */
  void reserve_resources(InnerTask *task);

  /*!
  * @brief Create, assign dependencies, and enqueue data movement tasks for the
  task.
  * @param task The task to create data movement tasks for.
  */
  void create_datamove_tasks(InnerTask *task);
};

/**
 * @brief RuntimeReserver phase of the scheduler.
 * @details RuntimeReserver reserves all 'non-persistent resources`.
 * Here all 'non-persistent resources` that
 * have a lifetime equal to the task body are reserved and are not directly
 * shared between tasks. At the moment this is only the VCUS/Threads a task
 * uses. This phase submits the task to the launcher.
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
  /// The multidevice queues for compute tasks
  std::shared_ptr<PhaseManager<ResourceCategory::NonPersistent>> runnable_tasks;
  /// The multidevice queues for data movement tasks
  std::shared_ptr<PhaseManager<ResourceCategory::Movement>> movement_tasks;

  /// The name of the phase. Used for debugging and tracing.
  inline static const std::string name{"Runtime Reserver"};
  /// The status of the phase. Used for debugging and tracing.
  RuntimeReserverStatus status{name};
  /// The buffer of tasks enqueued (and waiting) in the phase.
  std::vector<InnerTask *> launchable_tasks_buffer;

  /*!
  @brief Check if the NonPersistentResources are available for a ComputeTask.
  */
  bool check_resources(InnerTask *task);

  /*!
   * @brief Check the resources for a data movement task (number of free copy
   * engines).
   */
  bool check_data_resources(InnerTask *task);

  /*!
   * @brief Reserve (decrease) the NonPersistentResources for a ComputeTask.
   */
  void reserve_resources(InnerTask *task);

  /*!
   * @brief Reserve (decrease) the resources for a data movement task (number of
   * copy engines).
   */
  void reserve_data_resources(InnerTask *task);
};

/*!
 * @brief The Launcher phase of the scheduler. Dispatches tasks to workers.
 */
class Launcher : virtual public SchedulerPhase {
public:
  /*Number of running tasks. A task is running if it has been assigned to a
   * worker and is not complete*/
  std::atomic<size_t> num_running_tasks{0};

  Launcher(InnerScheduler *scheduler, DeviceManager *devices)
      : SchedulerPhase(scheduler, devices) {}

  /// Add a task to the launcher. Assigns a task to a free worker.
  void enqueue(InnerTask *task){};
  /// Add a task to the launcher. Assigns a task to a free worker.
  void enqueue(InnerTask *task, InnerWorker *worker);
  /// Add a batch of tasks to the launcher. Assigns them to free workers.
  void enqueue(std::vector<InnerTask *> &tasks){};

  /// A placeholder function in case work needs to be done at this stage. Not
  /// used.
  void run();
  void run(SchedulerPhase *next_phase) { this->run(); };

  /// Number of running tasks. A task is running if it has been assigned to a
  /// worker and is not complete
  size_t get_count() { return this->num_running_tasks.load(); }

protected:
  /// The name of the phase. Used for debugging and tracing.
  inline static const std::string name{"Launcher"};
  /// The status of the phase. Used for debugging and tracing.
  LauncherStatus status{name};
  /// Buffer to store not yet launched tasks. Currently unused.
  TaskList task_buffer;
  /// Buffer to store unassigned workers. Currently unused.
  WorkerList worker_buffer;
};

#endif // PARLA_PHASES_HPP
