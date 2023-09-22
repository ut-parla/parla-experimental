#pragma once
#ifndef PARLA_BACKEND_HPP
#define PARLA_BACKEND_HPP

/**
 * @mainpage Parla Documentation
 *
 * Welcome to the core C++ & Cython documentation for Parla.
 * This is the landing page for the Doxygen-generated HTML documentation
 * including call graphs and inheritance diagrams. You likely got here from the
 * MKDocs documentation, which is the main user-facing documentation for Parla.
 * This page exists to help contributors navigate the C++ runtime.
 *
 * @section sec_links Links
 *
 * - [GitHub Repository](https://github.com/ut-parla/parla-experimental)
 */

/*! @file runtime.hpp
 *  @brief The core C++ runtime for Parla. Includes the main scheduler and task
 * classes.
 */

#include "device.hpp"
#include "resources.hpp"
#include <algorithm>
#include <assert.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "containers.hpp"

#include "device_manager.hpp"
#include "gpu_utility.hpp"
#include "parray.hpp"
#include "parray_tracker.hpp"
#include "profiling.hpp"
#include "resource_requirements.hpp"
#include "memory_manager.hpp"

using namespace std::chrono_literals;
using namespace parray;

// General Note: A LOT of these atomics could just be declared as volatile.

// Forward Declarations of Inner Classes
class InnerTask;
class TaskBarrier;
class InnerWorker;
class InnerScheduler;

class RLTaskMappingPolicy;

// Type Aliases for common containers
using WorkerQueue = ProtectedQueue<InnerWorker *>;
using WorkerList = ProtectedVector<InnerWorker *>;
using TaskQueue = ProtectedQueue<InnerTask *>;
using TaskList = ProtectedVector<InnerTask *>;
using SpaceList = ProtectedVector<TaskBarrier *>;
using PointerList = ProtectedVector<uintptr_t>;
using CreatedPArrayQueue = ProtectedQueue<PArrayLocationSize_t>;

using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

/// Global New PArray List for arrays instatiated outside of the runtime
inline CreatedPArrayQueue unmapped_created_parrays;
inline CreatedPArrayQueue unreserved_created_parrays;

// Callbacks into Python

typedef void (*launchfunc_t)(void *scheduler, void *task, void *worker);
typedef void (*stopfunc_t)(void *scheduler);

/*!
 * @brief Callback function to launch a single task by calling into the Python
 * scheduler.
 * @note Currently unused in the C++ runtime. Legacy implementation from the
 * first C++ runtime which would acquire the GIL from the scheduler thread.
 */
inline void launch_task_callback(launchfunc_t func, void *scheduler, void *task,
                                 void *worker) {
  func(scheduler, task, worker);
}

/*!
 * @brief Callback function to stop the main scheduler thread. Acquires the
 * interpreter and signals the Python runtime to stop.
 * @note Called at Parla runtime shutdown (at context destruction or during
 * exception handling)
 */
inline void launch_stop_callback(stopfunc_t func, void *scheduler) {
  func(scheduler);
}

enum class MappingPolicyType {
  LoadBalancingLocality = 0,
  RLTraining = 1,
  RLTest = 2
};

inline void add_unmapped_created_parray(InnerPArray *parray, DevID_t dev_id,
                                        size_t to_move_size) {
  PArrayLocationSize_t parray_location =
      std::make_tuple(parray, dev_id, to_move_size);
  unmapped_created_parrays.push_back(parray_location);
}

inline void add_unreserved_created_parray(InnerPArray *parray, DevID_t dev_id,
                                          size_t to_move_size) {

  PArrayLocationSize_t parray_location =
      std::make_tuple(parray, dev_id, to_move_size);
  unreserved_created_parrays.push_back(parray_location);
}

inline void create_parray(InnerPArray *parray, int parray_device_id) {
  auto dev_id = parrayid_to_globalid(parray_device_id);
  size_t to_move_size = parray->get_size();
  add_unmapped_created_parray(parray, dev_id, to_move_size);
  add_unreserved_created_parray(parray, dev_id, to_move_size);
}

/*!
 * @brief Tracks the state of of a task within the runtime
 * @note  This is the lifecycle of a task within the runtime. These states
 * depend on the runtime acting on the task, in contrast to the Status of a task
 * which depends on the state of the tasks dependencies.
 */
enum class TaskState {
  /// Initial State. Task has been created but not spawned
  CREATED = 0,
  /// Task has been spawned
  SPAWNED = 1,
  /// Task has been mapped
  MAPPED = 2,
  /// Task has persistent resources reserved
  RESERVED = 3,
  /// Task is ready to run
  READY = 4,
  /// Task is currently running and has runtime resources reserved
  RUNNING = 5,
  /// Task body has completed but GPU kernels may be asynchronously running
  RUNAHEAD = 6,
  /// Task has completed
  COMPLETED = 7
};

/*!
 * @brief The type of between task synchronization to use in runahead scheduling
 * on device hardware queues
 */
enum class SynchronizationType {
  /// No unahead scheduling. Tasks block for body completion before running
  /// ahead
  NONE = 0,
  /// Block task body execution by waiting for events on the streams from
  /// dependency tasks to complete
  BLOCKING = 1,
  /// Do not block task body execution. Cross stream wait events are added to
  /// the tasks streams before the body executes.
  NON_BLOCKING = 2,
  /// No synchronization (relies on user written code to ensure state)
  USER = 3
};

/*
 * @brief Struct to store task status
 */
class TaskStatusFlags {
public:
  bool spawnable{false};
  bool mappable{false};
  bool reservable{false};
  bool compute_runnable{false};
  bool runnable{false};

  TaskStatusFlags() = default;

  TaskStatusFlags(bool spawnable, bool mappable, bool reservable,
                  bool compute_runnable, bool runnable)
      : spawnable(spawnable), mappable(mappable), reservable(reservable),
        compute_runnable(compute_runnable), runnable(runnable) {}

  bool any() {
    return spawnable || mappable || reservable || compute_runnable || runnable;
  }
};

/* Properties of the tasks dependencies */
enum class TaskStatus {
  // Initial State. Status of dependencies is unknown or not spawned
  INITIAL = 0,
  // All dependencies are spawned (this task can be safely spawned)
  SPAWNABLE = 1,
  // All dependencies are mapped (this task can be safely mapped)
  MAPPABLE = 2,
  // All dependencies have persistent resources reserved (this task can be
  // safely reserved)
  RESERVABLE = 3,
  // All compute dependencies have RUNAHEAD/COMPLETED status
  COMPUTE_RUNNABLE = 4,
  // All (including data) dependencies have RUNAHEAD/COMPLETED status
  RUNNABLE = 5
};

#ifdef PARLA_ENABLE_LOGGING
BINLOG_ADAPT_STRUCT(TaskStatusFlags, spawnable, mappable, reservable,
                    compute_runnable, runnable)
BINLOG_ADAPT_ENUM(TaskState, CREATED, SPAWNED, MAPPED, RESERVED, READY, RUNNING,
                  RUNAHEAD, COMPLETED)
BINLOG_ADAPT_ENUM(TaskStatus, INITIAL, SPAWNABLE, MAPPABLE, RESERVABLE,
                  COMPUTE_RUNNABLE, RUNNABLE)
#endif

/// @brief A pair of a task and its status information
using TaskStatusPair = std::pair<InnerTask *, TaskStatusFlags>;

/// @brief A list of task status pairs
using TaskStatusList = std::vector<TaskStatusPair>;

/**
 *   @brief The C++ runtime implementation of a task.
 *   Inherits metadata from the Python layer.
 */
class InnerTask {

  // TODO(hc): those member vars should be protected.
public:
  size_t approximated_num_siblings{0};

  // Task info. for RL
  double remote_data_bytes{0};
  double num_dependencies{0};
  double num_dependents{0};

  size_t mapping_order_id;
  size_t launching_order_id;

  /* Unique ID of the task. Can be used as a dictionary key.*/
  long long int id = 0;

  size_t replay_mem_buffer_id_ = 0;

  /*Name of the task. Useful for logging and printing.*/
  std::string name = "";

  /*Instance count of the task (Number of continuations of this task)*/
  int instance = 0;

  /* State of the task (where is this task)*/
  std::atomic<TaskState> state{TaskState::CREATED};

  /* Status of the task (state of its dependencies)*/
  std::atomic<TaskStatus> status{TaskStatus::INITIAL};

  /* Reference to the scheduler (used for synchronizing state on events) */
  InnerScheduler *scheduler = nullptr;

  /*Container for Events*/
  PointerList events;

  /*Synchronization Type */
  SynchronizationType sync_type = SynchronizationType::NON_BLOCKING;

  /*Container for Streams*/
  PointerList streams;

  /*Task monitor*/
  std::mutex mtx;

  /* Priority of the task. Higher priority tasks are scheduled first. */
  std::atomic<int> priority{0};

  /* The pointer to the Python Task which contains the class body */
  void *py_task = nullptr; // TODO: Refactor to PyObject type?

  /* Container of Task Dependencies (should be thread-safe)*/
  TaskList dependencies;

  /* Container of Task Dependents (should be thread-safe)*/
  TaskList dependents;

  /* Container of Task Spaces */
  SpaceList spaces;

  /*Local depdendency buffer*/
  std::vector<InnerTask *> dependency_buffer = std::vector<InnerTask *>();

  /* Number of blocking (uncompleted) compute task dependencies */
  std::atomic<int> num_blocking_compute_dependencies{1};

  /* Number of  blocking (uncompleted) task (compute+data) dependencies */
  std::atomic<int> num_blocking_dependencies{1};

  /* Number of unspawned dependencies */
  std::atomic<int> num_unspawned_dependencies{1};

  /* Number of unmapped dependencies */
  std::atomic<int> num_unmapped_dependencies{1};

  /* Number of unreserved dependencies */
  std::atomic<int> num_unreserved_dependencies{1};

  /*Number of unreserved instances (for multidevice) */
  std::atomic<int> num_persistant_instances{1};
  bool removed_reserved{false};

  /* Number of waiting instances (for multidevice) */
  std::atomic<int> num_runtime_instances{1};
  bool removed_runtime{false};

  /* Task Assigned Device Set*/
  std::vector<ParlaDevice *> assigned_devices;

  /*Resource Requirements for each assigned device*/
  std::unordered_map<int, ResourcePool_t> device_constraints;

  /* Task is data movement task */
  std::atomic<bool> is_data{false};

  /* Task has processed data into data tasks (if any exists). Defaults to true
   * if none exist. */
  std::atomic<bool> processed_data{true};

  /* A list of a pair of PArray instances and access modes to them.
     The first dimension index is for a device id specified in @spawn.
     The second index space is for PArrays. */
  std::vector<std::vector<PArrayAccess_t>> parray_list;

  // Epochs from the first epoch to task launching
  double begin_time_epochs{0};
  // Epochs from the first epoch to task mapping 
  double mapping_time_epochs{0};
  // Epochs from the first epoch to task completion
  double completion_time_epochs{0};
  // Max completion time among dependency tasks
  double max_depcompl_time_epochs{0};

  InnerTask();
  InnerTask(long long int id, void *py_task);
  InnerTask(std::string name, long long int id, void *py_task);

  /* Set the scheduler */
  void set_scheduler(InnerScheduler *scheduler);

  /* Set the name of the task */
  void set_name(std::string name);

  /* Get the name of the task */
  const std::string &get_name() const { return this->name; };

  /* Set the id of the task */
  void set_id(long long int name);

  /* Set the python task */
  void set_py_task(void *py_task);

  /* Set the priority of the task */
  void set_priority(int priority);

  /* Add a dependency to the task buffer but don't process it*/
  void queue_dependency(InnerTask *task);

  /* Add a list of dependencies to the task. For external use.*/
  TaskStatusFlags process_dependencies();

  /* Clear the dependency list */
  void clear_dependencies();

  /* Add a dependency to the task and process it*/
  TaskState add_dependency(InnerTask *task);

  /* Add a list of dependencies to the task and process them. For external
   * use.*/
  TaskStatusFlags add_dependencies(std::vector<InnerTask *> &tasks,
                                   bool data_tasks = false);

  /* Add a dependent to the task */
  TaskState add_dependent_task(InnerTask *task);
  TaskState add_dependent_space(TaskBarrier *barrier);

  /* Add a list of dependents to the task */
  // void add_dependents(std::vector<bool> result, std::vector<InnerTask*>&
  // tasks);

  /* Handle the creation of a new parray object */
  void create_parray(InnerPArray *parray, int parray_device_id);
  /*
   * Add a PArray to the task
   *
   * @param parray Pointer to a PArray that this task use
   * @param access_mode Access mode TODO(hc): This type is int and
   *                                          it is immediately casted to
   *                                          an enum type. This function
   *                                          is called by Python through
   * Cython, but C++ enum and Python enum or int are not compatible. So, for
   * conveniency, I just pass int between Python and C++.
   */
  void add_parray(InnerPArray *parray, int access_mode, int dev_id);

  /*
   *  Notify dependents that dependencies have completed
   *  This should be called by the worker when a task has completed
   *  Returns a container of tasks that are now ready to run
   *  TODO: Decide on a container to use for this
   */
  void notify_dependents(TaskStatusList &tasks, TaskState new_state);
  void notify_dependents_completed();

  /* Wrapper for testing */
  bool notify_dependents_wrapper();

  /*
   *  Notify the task that one of its dependents has completed
   *  Decrements the number of blocking dependencies.
   *  Return true if 0 blocking dependencies remain.
   *  Used by "notify_dependents"
   */
  TaskStatusFlags notify(TaskState dependency_state, bool is_data = false);

  void get_dependency_completion_epochs(double dep_compltime_epochs);

  /* Reset state and increment all internal counters. Used by continuation */
  void reset() {
    // TODO(wlr): Should this be done with set_state and assert old==RUNNING?
    this->state.store(TaskState::SPAWNED);
    this->status.store(TaskStatus::INITIAL);
    this->instance++;
    this->num_blocking_compute_dependencies.store(1);
    this->num_blocking_dependencies.store(1);
    this->num_unspawned_dependencies.store(1);
    this->num_unmapped_dependencies.store(1);
    this->num_unreserved_dependencies.store(1);
    this->assigned_devices.clear();
    // this->reset_events_streams();
  }

  /* Return whether the task is ready to run */
  bool blocked();

  /* Get a task name */
  std::string get_name();

  /* Return True if an instance is a data movement task */
  const bool is_data_task() const {
    return this->is_data.load(std::memory_order_relaxed);
  }

  /* Get number of dependencies */
  int get_num_dependencies();

  /* Get number of dependents */
  int get_num_dependents();

  /* Get number of blocking dependencies */
  inline int get_num_blocking_dependencies() const {
    return this->num_blocking_dependencies.load();
  };

  inline int get_num_unmapped_dependencies() const {
    return this->num_unmapped_dependencies.load();
  };

  template <typename ResourceCategory> inline void set_num_instances() {
    if constexpr (std::is_same<ResourceCategory,
                               Resource::PersistentResources>::value) {
      this->num_persistant_instances.store(this->assigned_devices.size());
    } else {
      this->num_runtime_instances.store(this->assigned_devices.size());
    }
  };

  template <typename ResourceCategory> inline int decrement_num_instances() {
    if constexpr (std::is_same<ResourceCategory,
                               Resource::PersistentResources>::value) {
      return this->num_persistant_instances.fetch_sub(1);
    } else {
      return this->num_runtime_instances.fetch_sub(1);
    }
  };

  template <typename ResourceCategory> inline int get_num_instances() {
    if constexpr (std::is_same<ResourceCategory,
                               Resource::PersistentResources>::value) {
      return this->num_persistant_instances.load();
    } else {
      return this->num_runtime_instances.load();
    }
  };

  template <typename ResourceCategory> inline bool get_removed() {
    if constexpr (std::is_same<ResourceCategory,
                               Resource::PersistentResources>::value) {
      return this->removed_reserved;
    } else {
      return this->removed_runtime;
    }
  }

  template <typename ResourceCategory> inline void set_removed(bool waiting) {
    if constexpr (std::is_same<ResourceCategory,
                               Resource::PersistentResources>::value) {
      this->removed_reserved = waiting;
    } else {
      this->removed_runtime = waiting;
    }
  }

  /* Get dependency list. Used for testing Python interface. */
  std::vector<void *> get_dependencies();

  /* Get dependents list. Used for testing Python interface. */
  std::vector<void *> get_dependents();

  /*Add event to task*/
  void add_event(uintptr_t event) { this->events.push_back(event); }

  /*Add stream to task */
  void add_stream(uintptr_t stream) { this->streams.push_back(stream); };

  /* Reset events and streams */
  void reset_events_streams() {
    this->events.clear();
    this->streams.clear();
  }

  /* Synchronize self */
  void synchronize_events() {
    size_t num_events = this->events.size_unsafe();
    for (size_t i = 0; i < num_events; i++) {
      uintptr_t event_ptr = this->events.at_unsafe(i);
      event_synchronize(event_ptr);
    }
  }

  /*!
   * @brief Dispatches to the appropriate synchronization function for runahead
   * scheduling
   * @param sync_type SynchronizationType::BLOCKING or
   * SynchronizationType::NON_BLOCKING
   */
  void handle_runahead_dependencies(int sync_type_int) {
    SynchronizationType sync_type =
        static_cast<SynchronizationType>(sync_type_int);
    if (sync_type == SynchronizationType::BLOCKING) {
      this->synchronize_dependency_events();
    } else if (sync_type == SynchronizationType::NON_BLOCKING) {
      this->wait_dependency_events();
    }
  }

  /*Synchronize dependencies*/
  void synchronize_dependency_events() {
    size_t num_dependencies = this->dependencies.size_unsafe();
    for (size_t i = 0; i < num_dependencies; i++) {
      InnerTask *dependency = this->dependencies.at_unsafe(i);
      dependency->synchronize_events();
    }
  }

  /*Wait dependencies*/
  // TODO(wlr): This locking is overkill. Some of these aren't even necessary.
  // Comment(wlr): Removing all locks. By the time this executes all
  // dependencies will have ran their task bodies (can assume no more
  // modifications)
  void wait_dependency_events() {

    std::cout << "Setting wait triggers for dependencies of "
              << this->get_name() << std::endl;

    // For each dependency, wait on all of its events on all of our streams
    size_t num_dependencies = this->dependencies.size_unsafe();
    for (size_t i = 0; i < num_dependencies; i++) {
      InnerTask *dependency = this->dependencies.at_unsafe(i);
      auto &dependency_events = dependency->events;

      std::cout << "Waiting for event from dependency: "
                << dependency->get_name() << std::endl;
      size_t num_events = dependency_events.size_unsafe();
      for (size_t j = 0; j < num_events; j++) {
        uintptr_t event_ptr = dependency_events.at_unsafe(j);
        // Wait on the event on all of our streams
        size_t num_streams = this->streams.size_unsafe();
        for (size_t k = 0; k < num_streams; k++) {
          uintptr_t stream_ptr = this->streams.at_unsafe(k);
          event_wait(event_ptr, stream_ptr);
        }
      }
    }
  }

  /* Get python task */
  void *get_py_task();

  /* Get the python assigned devices */
  std::vector<ParlaDevice *> &get_assigned_devices();

  /*Add to the assigned device list*/
  void add_assigned_device(ParlaDevice *device);

  /* Fix the assigned device set.*/
  void finalize_assigned_devices();

  /*
   * Copy a vector of device pointers
   *
   * @param others Source vector of device pointers to copy
   */
  void copy_assigned_devices(const std::vector<ParlaDevice *> &others);

  /* Set the task status */
  int set_state(int state);

  /* Set the task state */
  TaskState set_state(TaskState state);

  /* Get the task state */
  TaskState get_state() const {
    const TaskState state = this->state.load();
    return state;
  }

  /*Set the task status */
  TaskStatus set_status(TaskStatus status);

  /*Determine status from parts*/
  // TODO(wlr): this should be private
  TaskStatus determine_status(bool spawnable, bool mappable, bool reservable,
                              bool ready);

  /*Get the task status*/
  TaskStatus get_status() const {
    const TaskStatus status = this->status.load();
    return status;
  }

  /* Set complete */
  void set_complete();

  /* Get complete */
  bool get_complete();

  void add_device_req(ParlaDevice *dev_ptr, MemorySz_t mem_sz, VCU_t num_vcus);
  void begin_arch_req_addition();
  void end_arch_req_addition();
  void begin_multidev_req_addition();
  void end_multidev_req_addition();

  PlacementRequirementCollections &get_placement_req_options() {
    return placement_req_options_;
  }

  /* Set epochs at task launching from when the task graph begins */
  void record_task_begin_epochs(double epochs) {
    this->begin_time_epochs = epochs;
  }

  /* Set epochs at task completion from when the task graph begins */
  void record_task_completion_epochs(double epochs) {
    this->completion_time_epochs = epochs;
  }

protected:
  /*
   *  1 <--> 3 (MultiDevAdd, normally SingleDevAdd) <--> 2*2 (SingleArchAdd)
   *  1 <--> 2 (SingleArchAdd)
   */
  enum ReqAdditionState {
    SingleDevAdd = 1,
    /* SingleArchAdd == 2, */
    MultiDevAdd = 3
  };
  uint32_t req_addition_mode_;
  std::shared_ptr<ArchitectureRequirement> tmp_arch_req_;
  std::shared_ptr<MultiDeviceRequirements> tmp_multdev_reqs_;
  // TODO(hc): rename these..
  PlacementRequirementCollections placement_req_options_;
};

class InnerDataTask : public InnerTask {
public:
  InnerDataTask() = delete;
  // TODO(hc): this id is not unique (In case of compute task,
  //           The Python runtime maintains the unique id and assigns it.
  //           but this data move task is created in C++ and we cannot
  //           immediately assign the unique id. We may need another function
  //           call from Python t C++ when we create Python data move task
  //           later. The current id for all the data move tasks is 0.
  InnerDataTask(std::string name, long long int id, InnerPArray *parray,
                AccessMode access_mode, int dev_id)
      : parray_(parray), access_mode_(access_mode), dev_id_(dev_id),
        InnerTask(name, id, nullptr) {
    this->is_data = true;
    // Data tasks are created after persistent resource reservation.
    // Therefore its start state is always RESERVED.
    this->set_state(TaskState::RESERVED);
  }

  /// Return a python PArray pointer (as void*).
  void *get_py_parray();

  /// Return a access mode of PArray (as int value, used for Python interface)
  int get_access_mode();

  // TODO(hc): will be removed
  int get_device_id() { return this->dev_id_; }

private:
  InnerPArray *parray_;
  AccessMode access_mode_;
  int dev_id_;
};

#ifdef PARLA_ENABLE_LOGGING
LOG_ADAPT_STRUCT(InnerTask, name, instance, get_state, get_status, is_data_task)
LOG_ADAPT_DERIVED(InnerDataTask, (InnerTask))
#endif

/**
 *  @brief A task barrier is a synchronization primitive that notifies when a
 * set of tasks are completed.
 */
class TaskBarrier {
  // TODO: As is, this is not resuable.

  // TODO: This assumes the Python holder of the TaskBarrier will not be deleted
  // before all of its tasks are completed. Otherwise its reference will be
  // cleaned by the GC and lead to a segfault. This is a serious problem. Add
  // backlinks for cleanup? How to handlw without a huge performance hit?

public:
  std::mutex mtx;
  std::condition_variable cv;
  int64_t id;

  std::atomic<int> num_incomplete_tasks{0};

  TaskBarrier() = default;

  TaskBarrier(int num_tasks) : num_incomplete_tasks(num_tasks) {}

  TaskState _add_task(InnerTask *task);
  void add_task(InnerTask *task);
  void add_tasks(std::vector<InnerTask *> &tasks);
  void set_id(int64_t id) { this->id = id; }

  void wait() {
    // std::cout << "Barrier Wait" << std::endl;
    std::unique_lock<std::mutex> lck(mtx);
    cv.wait(lck, [this] { return num_incomplete_tasks == 0; });
  }

  void notify() {
    std::unique_lock<std::mutex> lck(mtx);
    int prev = this->num_incomplete_tasks.fetch_sub(1);
    if (prev == 1) {
      cv.notify_all();
    }
  }
};

/*!
 * @brief The C++ backend for the Parla TaskSpace
 * @details The TaskSpace is a collection of tasks that can be queried (sliced)
 * to get a subset of tasks and synchronized on.
 */
class InnerTaskSpace : public TaskBarrier {

public:
  InnerTaskSpace() = default;

  std::unordered_map<int64_t, InnerTask *> task_map;

  void add_task(int64_t key, InnerTask *task) {
    task_map.insert({key, task});
    TaskBarrier::add_task(task);
  }

  void add_tasks(std::vector<int64_t> &keys, std::vector<InnerTask *> &tasks) {
    for (int i = 0; i < keys.size(); i++) {
      task_map.insert({keys[i], tasks[i]});
    }
    TaskBarrier::add_tasks(tasks);
  }

  void get_tasks(std::vector<int64_t> &keys, std::vector<InnerTask *> &tasks) {
    for (int i = 0; i < keys.size(); i++) {
      tasks.push_back(task_map[keys[i]]);
    }
  }

  void wait() { TaskBarrier::wait(); }

  void notify() { TaskBarrier::notify(); }
};

/**
 * @brief A worker is a thread that executes tasks.
 * @details The worker is a C++ object that is created by the Python runtime.
 * It is responsible for executing tasks assigned to it by the scheduler. The
 * worker sleeps until it receives a task from the scheduler. When it recieves a
 * task, it wakes up and acquires the GIL.
 */
class InnerWorker {

public:
  /* Pointer to Python Worker object */
  void *py_worker = nullptr;

  /* Pointer to the active task */
  // void* py_task = nullptr;
  InnerTask *task = nullptr;

  InnerScheduler *scheduler = nullptr;

  std::mutex mtx;
  std::condition_variable cv;
  bool ready = false;
  bool notified = false;

  int thread_idx = -1;

  /// A list of newly ready tasks that will be enqueued by this worker.
  TaskStatusList enqueue_buffer;

  // TODO: (improvement?) Custom Barrier and Event Handling

  // TODO: (improvement?) A buffer for multiple tasks assigned to a worker

  InnerWorker() = default;
  InnerWorker(void *worker) : py_worker(worker){};

  /// Set the backlink to the python worker
  void set_py_worker(void *worker) { this->py_worker = worker; };

  /// Store the scheduler that owns this worker
  void set_scheduler(InnerScheduler *scheduler) {
    this->scheduler = scheduler;
  };

  /// Set the thread index of this worker
  void set_thread_idx(int idx) { this->thread_idx = idx; };

  /// wait for a task to be assigned
  void wait();

  /// assign a task to the worker and notify worker that it is available
  void assign_task(InnerTask *task);

  /**
   * @brief Get the C++ task instance that this worker thread will execute.
   * This function returns two outputs, a pointer to a task pointer and
   * a pointer to a flag specifying if this task is data task or not.
   * If that is the data task, the callee creates a Python data task instance
   * and makes a connection between the Python and the C++ instances.
   *
   * @param task A pointer to a pointer to a task (output)
   * @param is_data_task A pointer to a flag that sets True if the task is data
   * task.
   */
  void get_task(InnerTask **task, bool *is_data_task);

  /* Remove task */
  void remove_task();

  void stop();

  /* Set epochs at task launching from when the task graph begins */
  void record_task_begin_epochs();

  /* Set epochs at task completion from when the task graph begins */
  void record_task_completion_epochs();
};

#ifdef PARLA_ENABLE_LOGGING
LOG_ADAPT_STRUCT(InnerWorker, thread_idx, notified)
#endif

/**
 * @brief A worker pool is a collection of workers.
 * @details For worker creation and managing status.
 * @tparam AllWorkers_t A container type for storing the list of all workers
 * (should be thread-safe)
 * @tparam ActiveWorkers_t A container type for storing the list of active
 * workers (should be thread-safe)
 */
template <typename AllWorkers_t, typename ActiveWorkers_t> class WorkerPool {

public:
  /// Container of all workers
  AllWorkers_t all_workers;

  /// Container of available workers
  ActiveWorkers_t active_workers;

  int max_workers;

  /// mutex for blocking additional spawn/await*/ until workers are notified
  std::mutex mtx;

  /// condition variable for blocking spawn/await*/
  std::condition_variable cv;

  /// number of notified but not running workers (waiting)
  std::atomic<int> notified_workers{0};

  WorkerPool() = default;
  WorkerPool(int nworkers) : max_workers(nworkers){};

  /// add a worker to the active pool
  void enqueue_worker(InnerWorker *worker);

  /// remove a worker from the active pool
  InnerWorker *dequeue_worker();

  /// add a worker to the all pool
  void add_worker(InnerWorker *worker);

  /// get number of available workers
  int get_num_available_workers();

  /// get number of total workers
  int get_num_workers();

  /// set number of total workers
  void set_num_workers(int nworkers);

  /// increase number of notified workers
  int increase_num_notified_workers();

  // decrease number of notified workers
  int decrease_num_notified_workers();

  // get number of notified workers
  inline int get_num_notified_workers() {
    return this->notified_workers.load();
  }

  /// @brief barrier to block additional for spawns so that other waiting
  /// workers threads can take the GIL*/
  void spawn_wait();

  /// Remove a worker from the all pool
  // void remove_worker(InnerWorker* worker);
};

typedef WorkerPool<WorkerQueue, WorkerQueue> WorkerPool_t;

// Forward declaration of scheduler phases

class Mapper;
class MemoryReserver;
class RuntimeReserver;
class Launcher;
class MappingPolicy;
class LocalityLoadBalancingMappingPolicy;

namespace Scheduler {

enum State {
  spawned,
  mapped,
  reserved,
  ready,
  launch,
  running,
  complete,
  failed
};

class Status {
private:
  const static int size = 8;

public:
  int status[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  void reset() {
    for (int i = 0; i < size; i++) {
      status[i] = 0;
    }
  }

  void set(int index, int value) { this->status[index] = value; }

  int get(int index) { return this->status[index]; }

  void update(State state) { this->status[state]++; }

  void print() {
    std::cout << "Scheduler Status: (" << this->status[0] << ", "
              << this->status[1] << ", " << this->status[2] << ", "
              << this->status[3] << ", " << this->status[4] << ", "
              << this->status[5] << ", " << this->status[6] << ", "
              << this->status[7] << ")" << std::endl;
  }
};

} // namespace Scheduler

/**
 *   The C++ "Mirror" of Parla's Python Scheduler
 *   This class is used to create a C++ representation of a Parla Scheduler
 *   All scheduling logic should be handled by these after creation until
 * launched by the Python callback
 */
class InnerScheduler {

public:
  size_t task_mapping_log_ptr{0};
  size_t task_launching_log_ptr{0};
  // TODO(hc): not yet using these modes.
  /// 0 = disabled, 1 = replay decision (homogeneous tasks),
  /// 2 = replay decision and order.
  uint32_t task_mapping_log_mode{1};
  /// 0 = dsiabled, 1 = replay order.
  uint32_t task_launching_log_mode{0};
  bool task_mapping_log_registered{false};
  bool task_launching_log_registered{false};
  bool end_task_graph_is_last{false};
  std::vector<std::pair<std::string, DevID_t>> task_mapping_log;
  std::vector<std::string> task_launching_log;
  ProtectedUnorderedMap<std::string, InnerTask*> task_name_to_task;
  std::atomic<int> task_log_register_counter{0};
  size_t num_epochs{0};

  uint64_t task_mapping_order_{0};
  uint64_t task_launching_order_{0};

  double epoch_begin_epochs{0};
  double epoch_end_epochs{0};
  // in ms
  double previous_exec_time{std::numeric_limits<double>::max()};

  /* Sleep Between Loops */
  bool sleep_flag = false;

  /* Time to sleep between loops (microseconds) */
  int sleep_time = 20;

  /* Task Buffer */
  std::vector<InnerTask *> task_buffer = std::vector<InnerTask *>(10);

  /* Container of Thread Workers */
  WorkerPool_t workers;

  /* Active task counter (thread-safe) */
  std::atomic<int> num_active_tasks{1};

  /* Should Run, Stop Condition */
  std::atomic<bool> should_run = true;

  /* Phase: maps tasks to devices */
  Mapper *mapper;

  /* If it is set, break an infinite loop in InnerScheduler::run()
     and invoke PArray eviction from PythonScheduler */
  bool break_for_eviction = false;
  /* Memory size to evict for each device */
  std::vector<size_t> memory_size_to_evict{0};

  /* Track the number of eviction manager invocation */
  size_t num_eviction_invocation{0};

  /* Set necessary memory size getting from eviction manager
     on each device */
  void set_memory_size_to_evict(size_t, DevID_t);
  /* Get memory size to evict for each device */
  size_t get_memory_size_to_evict(DevID_t);

  /* Phase reserves resources to limit/plan task execution*/
  MemoryReserver *memory_reserver;
  RuntimeReserver *runtime_reserver;

  /*Responsible for launching a task. Signals worker thread*/
  Launcher *launcher;

  InnerScheduler(LRUGlobalEvictionManager *memory_manager,
      DeviceManager *device_manager, MappingPolicyType policy_type);
  ~InnerScheduler();
  // InnerScheduler(int nworkers);

  /* Pointer to callback to stop the Python scheduler */
  stopfunc_t stop_callback;

  /* Pointer to Python scheduler */
  void *py_scheduler;

  /* Scheduler Status */
  Scheduler::Status status;

  /* Set the number of workers */
  void set_num_workers(int nworkers);

  /* Set Python Scheduler */
  void set_py_scheduler(void *py_scheduler);

  /* Set Python "stop" callback */
  void set_stop_callback(stopfunc_t stop_callback);

  /* Get a flag that represents if there is still a task to be executed */
  bool get_should_run();

  /* Run the scheduler thread. Active for the lifetime of the Parla program */
  void run();

  /*Stop the scheduler. Called at the end of the Parla program */
  void stop();

  /* Activate scheduler on current thread. Runs through scheduler phases. */
  Scheduler::Status activate();

  /* Activate wrapper for Python layer (for use as scheduler callback) */
  void activate_wrapper();

  /*Spawn a Task (increment active, set state, possibly enqueue)*/
  void spawn_task(InnerTask *task);

  /* Enqueue task. */
  void enqueue_task(InnerTask *task, TaskStatusFlags flags);

  /* Enqueue more than one task */
  void enqueue_tasks(TaskStatusList &tasks);

  /* Add worker */
  void add_worker(InnerWorker *worker);

  /* Enqueue worker. */
  void enqueue_worker(InnerWorker *worker);

  /* Complete all task finalization. Notify Dependents / Release Resources /
   * Worker Enqueue */
  void task_cleanup(InnerWorker *worker, InnerTask *task, int state);

  void task_cleanup_presync(InnerWorker *worker, InnerTask *task, int state);
  void task_cleanup_postsync(InnerWorker *worker, InnerTask *task, int state);

  /* Get number of active tasks. A task is active if it is spawned but not
   * complete */
  int get_num_active_tasks();

  /* Increase number of active tasks */
  void increase_num_active_tasks();

  /* Decrease number of active tasks. If zero tasks are active, stop the
   * scheduler */
  void decrease_num_active_tasks();

  /*Increase number of notified workers*/
  int increase_num_notified_workers();

  /*Decrease number of notified workers*/
  int decrease_num_notified_workers();

  /* Get number of running tasks. A task is running if is Python task has been
   * assigned and the task is not complete*/
  int get_num_running_tasks();

  /* Get number of ready tasks. A task is ready if its dependencies has been
   * dispatched to a hardware queue or are complete */
  int get_num_ready_tasks();

  /* Get number of noitified workers */
  int get_num_notified_workers() {
    return this->workers.get_num_notified_workers();
  }

  /* Reserve a PArray in a device */
  void create_parray(InnerPArray *parray, int parray_device_id);

  /* Release a PArray in a device */
  void remove_parray(InnerPArray *parray, DevID_t global_dev_id);

  /* Reflect a PArray removal to PArray trackers */
  void remove_parray_from_tracker(
      parray::InnerPArray *parray, DevID_t global_dev_id);

  /* A task started to refer to the PArray, and so, reflect that to
   * the eviction manager (mm) */
  void grab_parray_reference(
      parray::InnerPArray *parray, DevID_t global_dev_id) {
    this->mm_->grab_parray_reference(parray, global_dev_id);
  }

  /* A task released the PArray, and so, reflect that to the eviction
   * manager (mm) */
  void release_parray_reference(
      parray::InnerPArray *parray, DevID_t global_dev_id) {
    this->mm_->release_parray_reference(parray, global_dev_id);
  }

  size_t get_mm_evictable_bytes(DevID_t dev_id) {
    return this->mm_->get_evictable_bytes(dev_id);
  }

  /* Get mapped memory on device */
  size_t get_mapped_memory(DevID_t global_dev_id);

  /* Get reserved memory on device */
  size_t get_reserved_memory(DevID_t global_dev_id);

  /*Get total memory on device */
  size_t get_max_memory(DevID_t global_dev_id);

  /* Get parray state in mapping parray tracker */
  bool get_mapped_parray_state(DevID_t global_dev_idx, uint64_t parray_id);

  /* Get parray state in reserved parray tracker */
  bool get_reserved_parray_state(DevID_t global_dev_idx, uint64_t parray_id);

  /* Spawn wait. Slow down the compute bound spawning thread so tasks on other
   * threads can start*/
  void spawn_wait();

  DeviceManager *get_device_manager() { return this->device_manager_; }

  TimePoint get_initial_epoch() {
    return this->device_manager_->get_initial_epoch();
  }

  void assign_task_mapping_id(InnerTask *task) {
    task->mapping_order_id = this->task_mapping_order_;
    //std::cout << "[Assigning Mapping ID] " << task->name <<
    //  ", " << task->mapping_order_id << "\n" << std::flush;
    ++this->task_mapping_order_;
  }

  void assign_task_launching_id(InnerTask *task) {
    task->launching_order_id = this->task_launching_order_;
    //std::cout << "[Assigning Launching ID] " << task->name <<
    //  ", " << task->launching_order_id << "\n" << std::flush;
    ++this->task_launching_order_;
  }

  void reset_task_mapping_id() {
    this->task_mapping_order_ = 0;
  }

  void reset_task_launching_id() {
    this->task_launching_order_ = 0;
  }

  void reset_task_name_to_task() {
    //InnerTask* global_task_instance = this->task_name_to_task.find("global_task_0")->second; 
    this->task_name_to_task.clear();
    //this->task_name_to_task.emplace("global_task_0", global_task_instance);
  }

  /// Increase the total number of tasks mapped to any device and
  /// to device did.
  ///
  /// @param dev_id device global id where a task is mapped
  /// @return the updated number of the tasks
  size_t atomic_incr_num_mapped_tasks_device(DevID_t dev_id,
                                             size_t weight = 1) {
    total_num_mapped_tasks_.fetch_add(weight, std::memory_order_relaxed);
    return dev_num_mapped_tasks_[dev_id].fetch_add(weight,
                                                   std::memory_order_relaxed);
  }

  /// Decrease the total number of tasks mapped to any device and
  /// to device did.
  ///
  /// @param dev_id device global id where a task is mapped
  /// @return the updated number of the tasks
  size_t atomic_decr_num_mapped_tasks_device(DevID_t dev_id,
                                             size_t weight = 1) {
    total_num_mapped_tasks_.fetch_sub(weight, std::memory_order_relaxed);
    return dev_num_mapped_tasks_[dev_id].fetch_sub(weight,
                                                   std::memory_order_relaxed);
  }

  /// Return the total number of mapped tasks to any device.
  ///
  /// @return The old number of total mapped tasks
  const size_t atomic_load_total_num_mapped_tasks() const {
    return total_num_mapped_tasks_.load(std::memory_order_relaxed);
  }

  /// Return the number of mapped tasks to a single device.
  ///
  /// @param dev_id Device global ID where a task is mapped
  /// @return The old number of the tasks mapped to a device
  const size_t atomic_load_dev_num_mapped_tasks_device(DevID_t dev_id) const {
    return dev_num_mapped_tasks_[dev_id].load(std::memory_order_relaxed);
  }

  /// Increase the number of tasks mapped to device did in a mapped state.
  ///
  /// @param dev_id device global id where a task is mapped
  /// @return the updated number of the tasks
  size_t atomic_incr_num_tasks_mapped_states(DevID_t did) {
    return dev_num_tasks_mapped_states_[did].fetch_add(
        1, std::memory_order_relaxed);
  }

  /// Decrease the number of tasks mapped to device did in a mapped state.
  ///
  /// @param dev_id Device global ID where a task is mapped
  /// @return The updated number of the tasks
  size_t atomic_decr_num_tasks_mapped_states(DevID_t did) {
    return dev_num_tasks_mapped_states_[did].fetch_sub(
        1, std::memory_order_relaxed);
  }

  /// Return the number of tasks mapped to device did in a mapped state.
  ///
  /// @return The number of tasks before increase 
  const size_t atomic_load_dev_num_tasks_mapped_states(DevID_t did) const {
    return dev_num_tasks_mapped_states_[did].load(
        std::memory_order_relaxed);
  }

  /// Increase the number of tasks mapped to device did in a
  /// resource reserved state.
  ///
  /// @param dev_id device global id where a task is mapped
  /// @return the updated number of the tasks
  size_t atomic_incr_num_tasks_resreserved_states(DevID_t did) {
    return dev_num_tasks_resreserved_states_[did].fetch_add(
        1, std::memory_order_relaxed);
  }

  /// Decrease the number of tasks mapped to device did in a
  /// resource reserved state.
  ///
  /// @param dev_id Device global ID where a task is mapped
  /// @return The updated number of the tasks
  size_t atomic_decr_num_tasks_resreserved_states(DevID_t did) {
    return dev_num_tasks_resreserved_states_[did].fetch_sub(
        1, std::memory_order_relaxed);
  }

  /// Return the number of tasks mapped to device did in a
  /// resource reserved state.
  ///
  /// @return The number of tasks before increase 
  const size_t atomic_load_dev_num_tasks_resreserved_states(DevID_t did) const {
    return dev_num_tasks_resreserved_states_[did].load(
        std::memory_order_relaxed);
  }

  /// Increase the number of ready tasks mapped to device did.
  ///
  /// @param dev_id device global id where a task is mapped
  /// @return the updated number of the tasks
  size_t atomic_incr_num_ready_tasks(DevID_t did) {
    return dev_num_ready_tasks_[did].fetch_add(
        1, std::memory_order_relaxed);
  }

  /// Decrease the number of ready tasks mapped to device did.
  ///
  /// @param dev_id Device global ID where a task is mapped
  /// @return The updated number of the tasks
  size_t atomic_decr_num_ready_tasks(DevID_t did) {
    return dev_num_ready_tasks_[did].fetch_sub(
        1, std::memory_order_relaxed);
  }

  /// Return the number of ready tasks mapped to device did.
  ///
  /// @return The number of tasks before increase 
  const size_t atomic_load_dev_num_ready_tasks(DevID_t did) const {
    return dev_num_ready_tasks_[did].load(
        std::memory_order_relaxed);
  }

  /// Increase the number of running tasks mapped to device did.
  ///
  /// @param dev_id device global id where a task is mapped
  /// @return the updated number of the tasks
  size_t atomic_incr_num_running_tasks(DevID_t did) {
    return dev_num_running_tasks_[did].fetch_add(1, std::memory_order_relaxed);
  }

  /// Decrease the number of runnig tasks mapped to device did.
  ///
  /// @param dev_id Device global ID where a task is mapped
  /// @return The updated number of the tasks
  size_t atomic_decr_num_running_tasks(DevID_t did) {
    return dev_num_running_tasks_[did].fetch_sub(1, std::memory_order_relaxed);
  }

  /// Return the number of running tasks on device did.
  ///
  /// @return The number of tasks before increase 
  const size_t atomic_load_dev_num_running_tasks(DevID_t did) const {
    return dev_num_running_tasks_[did].load(std::memory_order_relaxed);
  }

  /// Reset task mapping log pointer to 0.
  void reset_task_mapping_log_pointer() {
    this->task_mapping_log_ptr = 0;
  }
  void reset_task_launching_log_pointer() {
    this->task_launching_log_ptr = 0;
  }
  void complete_task_mapping_log() {
    std::lock_guard<std::mutex> guard(this->task_mapping_log_mtx_);
    this->task_mapping_log_registered = true;
  }
  bool is_task_mapping_log_registered() {
    std::lock_guard<std::mutex> guard(this->task_mapping_log_mtx_);
    return this->task_mapping_log_registered;
  }
  void remove_task_mapping_log() {
    std::lock_guard<std::mutex> guard(this->task_mapping_log_mtx_);
    this->task_mapping_log_registered = false;
  }
  void complete_task_launching_log() {
    std::lock_guard<std::mutex> guard(this->task_launching_log_mtx_);
    this->task_launching_log_registered = true;
  }
  bool is_task_launching_log_registered() {
    std::lock_guard<std::mutex> guard(this->task_launching_log_mtx_);
    return this->task_launching_log_registered;
  }
  void remove_task_launching_log() {
    std::lock_guard<std::mutex> guard(this->task_launching_log_mtx_);
    this->task_launching_log_registered = false;
  }
  void register_task_mapping_log(InnerTask* task) {
    if (task_log_register_counter.load() == 1) {
      this->task_mapping_log.push_back({
          task->name, task->assigned_devices[0]->get_global_id()});
    }
  }
  void register_task_launching_log(InnerTask* task) {
    this->task_launching_log.push_back(task->name);
  }

  void complete_task_order_logs(InnerTask* task);

  // Check if task mapping/launching order logging is enabled or not
  // If the current mode is no logging, skip all related function calls.
  // If the current mode is logging, from 2nd iteration, log x (user parameters)
  // + 2 iteration executions.
  // We only need to use this function for avoiding logging completion marking
  // since if the logging is not completed, the replay will be never run.
  bool check_task_mapping_order_mode(uint32_t term);

  bool check_task_launching_order_mode(uint32_t term);
protected:
  /// It manages all device instances in C++.
  /// This is destructed by the Cython scheduler.
  DeviceManager *device_manager_;

  /// The total number of tasks mapped to and running on the whole devices.
  std::atomic<size_t> total_num_mapped_tasks_{0};
  /// The total number of tasks mapped to and running on a single device.
  std::vector<CopyableAtomic<size_t>> dev_num_mapped_tasks_;
  std::vector<CopyableAtomic<size_t>> dev_num_tasks_mapped_states_;
  std::vector<CopyableAtomic<size_t>> dev_num_tasks_resreserved_states_;
  std::vector<CopyableAtomic<size_t>> dev_num_ready_tasks_;
  std::vector<CopyableAtomic<size_t>> dev_num_running_tasks_;
  std::mutex task_mapping_log_mtx_;
  std::mutex task_launching_log_mtx_;
  LRUGlobalEvictionManager *mm_;
};

#endif // PARLA_BACKEND_HPP
