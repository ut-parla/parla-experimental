#pragma once
#ifndef PARLA_BACKEND_HPP
#define PARLA_BACKEND_HPP

#include <assert.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

using namespace std::chrono_literals;

#include "containers.hpp"
#include "device_manager.hpp"
#include "profiling.hpp"

// General Note. A LOT of these atomics could just be declared as volatile.

// Forward Declarations of Inner Classes
class InnerTask;
class InnerWorker;
class InnerScheduler;

// Type Aliases for common containers

using WorkerQueue = ProtectedQueue<InnerWorker *>;
using WorkerList = ProtectedVector<InnerWorker *>;

using TaskQueue = ProtectedQueue<InnerTask *>;
using TaskList = ProtectedVector<InnerTask *>;

// Busy sleep for a given number of microseconds
inline void cpu_busy_sleep(unsigned int micro) {
  // compute_range r("sleep::busy", nvtx3::rgb{0, 127, 127});
  // int count = 0;
  auto block = std::chrono::microseconds(micro);
  auto time_start = std::chrono::high_resolution_clock::now();

  auto now = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(now - time_start);

  do {
    now = std::chrono::high_resolution_clock::now();
    elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(now - time_start);
  } while (elapsed.count() < micro);
}

// Forward declaration of python callbacks

/* Python function to assign a task to a worker */
typedef void (*launchfunc_t)(void *scheduler, void *task, void *worker);

/* Python function to stop the scheduler */
typedef void (*stopfunc_t)(void *scheduler);

// Callback Launchers

/* C++ -> Cython callback to launch a single task */
inline void launch_task_callback(launchfunc_t func, void *scheduler, void *task,
                                 void *worker) {
  func(scheduler, task, worker);
}

/* C*+ -> Cython callback to stop the main scheduler. Called at runtime exit. */
inline void launch_stop_callback(stopfunc_t func, void *scheduler) {
  func(scheduler);
}

/**
 *   The C++ backend of Parla's ResourcePool
 *   All scheduling logic should be handled by these after creation.
 */
template <typename T> class InnerResourcePool {
  /*TODO : This is a mock placeholder for now. Used to test the scheduler and
   * settle on API. Assumes setup happens sequentially and is externally
   * thread-safe. : Internal representation should be a map of strings to types.
   *      : This should be able to subtract, check, and add a resource to the
   * bool in an internally thread-safe manner. : This should be able to do the
   * above for all resources in another pool : This only handles a single type
   * of resource. Ideal should be able to handle multiple types like the Python
   * dictionary :( : (Needed) Multilevel dispatch for mixed types. e.g.
   * threads->int, vcus->rational. constexpr if? : (Optional) Be able to compare
   * pools of different types
   */
public:
  std::atomic<T> vcus{0};

  InnerResourcePool() = default;

  void set(std::string name, T value);

  T get(std::string name);

  // bool check(std::string name, int value);
  // bool check(std::vector<std::string> names, std::vector<int> values);
  template <typename J> bool check_greater(InnerResourcePool<J> &other);

  template <typename J> bool check_lesser(InnerResourcePool<J> &other);

  // bool increase(std::string name, T value);
  // bool increase(std::vector<std::string> names, std::vector<T> values);
  template <typename J> T increase(InnerResourcePool<J> &other);

  // bool decrease(std::string name, T value);
  // bool decrease(std::vector<std::string> names, std::vector<T> values);
  template <typename J> T decrease(InnerResourcePool<J> &other);
};

namespace Task {

/*State of the task. Shows which part of the runtime the task is in.*/
enum State {
  // Initial State. Task has been created but not spawned
  CREATED = 0,
  // Task has been spawned
  SPAWNED = 1,
  // Task has been mapped
  MAPPED = 2,
  // Task has persistent resources reserved
  RESERVED = 3,
  // Task is ready to run
  READY = 4,
  // Task is currently running and has runtime resources reserved
  RUNNING = 5,
  // Task body has completed but GPU kernels may be asynchronously running
  RUNAHEAD = 6,
  // Task has completed
  COMPLETED = 7
};

class StatusFlags {
public:
  bool spawnable{false};
  bool mappable{false};
  bool reservable{false};
  bool compute_runnable{false};
  bool runnable{false};

  StatusFlags() = default;

  StatusFlags(bool spawnable, bool mappable, bool reservable,
              bool compute_runnable, bool runnable)
      : spawnable(spawnable), mappable(mappable), reservable(reservable),
        compute_runnable(compute_runnable), runnable(runnable) {}

  bool any() {
    return spawnable || mappable || reservable || compute_runnable || runnable;
  }
};

/* Properties of the tasks dependencies */
enum Status {
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

} // namespace Task

#ifdef PARLA_ENABLE_LOGGING
BINLOG_ADAPT_STRUCT(Task::StatusFlags, spawnable, mappable, reservable,
                    compute_runnable, runnable)
BINLOG_ADAPT_ENUM(Task::State, CREATED, SPAWNED, MAPPED, RESERVED, READY,
                  RUNNING, RUNAHEAD, COMPLETED)
BINLOG_ADAPT_ENUM(Task::Status, INITIAL, SPAWNABLE, MAPPABLE, RESERVABLE,
                  COMPUTE_RUNNABLE, RUNNABLE)
#endif

using TaskState = std::pair<InnerTask *, Task::StatusFlags>;
using TaskStateList = std::vector<TaskState>;

/**
 *   The C++ "Mirror" of Parla's Python Tasks
 *   This class is used to create a C++ representation of a Parla Task
 *   All scheduling logic should be handled by these after creation until
 * launched by the Python callback
 */
class InnerTask {

public:
  /* Unique ID of the task. Can be used as a dictionary key.*/
  long long int id = 0;

  /*Name of the task. Useful for logging and printing.*/
  std::string name = "";

  /*Instance count of the task (Number of continuations of this task)*/
  int instance = 0;

  /* State of the task (where is this task)*/
  std::atomic<Task::State> state{Task::CREATED};

  /* Status of the task (state of its dependencies)*/
  std::atomic<Task::Status> status{Task::INITIAL};

  /* Reference to the scheduler (used for synchronizing state on events) */
  InnerScheduler *scheduler = nullptr;

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

  /* Tasks Internal Resource Pool. */
  InnerResourcePool<float> resources;

  /* Task has data to be moved */
  std::atomic<bool> has_data{false};

  /* Task has processed data into data tasks (if any exists). Defaults to true
   * if none exist. */
  std::atomic<bool> processed_data{true};

  InnerTask();
  InnerTask(long long int id, void *py_task);
  InnerTask(std::string name, long long int id, void *py_task);

  /* Set the scheduler */
  void set_scheduler(InnerScheduler *scheduler);

  /* Set the name of the task */
  void set_name(std::string name);

  /* Set the id of the task */
  void set_id(long long int name);

  /* Set the python task */
  void set_py_task(void *py_task);

  /* Set the priority of the task */
  void set_priority(int priority);

  /*Add a data arguments to Task (list of Parrays)*/
  void add_data(/*vector of cpp parray type*/) {
    this->has_data = true;
    this->processed_data = false;
  }

  /*Set a resource of the task*/
  void set_resources(std::string resource_name, float resource_value);

  /* Add a dependency to the task buffer but don't process it*/
  void queue_dependency(InnerTask *task);

  /* Add a list of dependencies to the task. For external use.*/
  Task::StatusFlags process_dependencies();

  /* Clear the dependency list */
  void clear_dependencies();

  /* Add a dependency to the task and process it*/
  Task::State add_dependency(InnerTask *task);

  /* Add a list of dependencies to the task and process them. For external
   * use.*/
  Task::StatusFlags add_dependencies(std::vector<InnerTask *> &tasks);

  /* Add a dependent to the task */
  Task::State add_dependent(InnerTask *task);

  /* Add a list of dependents to the task */
  // void add_dependents(std::vector<bool> result, std::vector<InnerTask*>&
  // tasks);

  /*
   *  Notify dependents that dependencies have completed
   *  This should be called by the worker when a task has completed
   *  Returns a container of tasks that are now ready to run
   *  TODO: Decide on a container to use for this
   */
  void notify_dependents(TaskStateList &tasks, Task::State new_state);

  /* Wrapper for testing */
  bool notify_dependents_wrapper();

  /* Notify the task that one of its dependents has completed
   *  Decrements the number of blocking dependencies.
   *  Return true if 0 blocking dependencies remain.
   *  Used by "notify_dependents"
   */
  Task::StatusFlags notify(Task::State dependency_state, bool is_data = false);

  /* Reset state and increment all internal counters. Used by continuation */
  void reset() {
    // TODO(wlr): Should this be done with set_state and assert old==RUNNING?
    this->state.store(Task::CREATED);
    this->status.store(Task::INITIAL);
    this->instance++;
    this->num_blocking_compute_dependencies.store(1);
    this->num_blocking_dependencies.store(1);
    this->num_unspawned_dependencies.store(1);
    this->num_unmapped_dependencies.store(1);
    this->num_unreserved_dependencies.store(1);
  }

  /* Return whether the task is ready to run */
  bool blocked();

  /* Get number of dependencies */
  int get_num_dependencies();

  /* Get number of dependents */
  int get_num_dependents();

  /* Get number of blocking dependencies */
  int get_num_blocking_dependencies() const;
  int get_num_unmapped_dependencies() const;

  /* Get dependency list. Used for testing Python interface. */
  std::vector<void *> get_dependencies();

  /* Get dependents list. Used for testing Python interface. */
  std::vector<void *> get_dependents();

  /* Get python task */
  void *get_py_task();

  /* Set the task status */
  int set_state(int state);

  /* Set the task state */
  Task::State set_state(Task::State state);

  /* Get the task state */
  Task::State get_state() const {
    const Task::State state = this->state.load();
    return state;
  }

  /*Set the task status */
  Task::Status set_status(Task::Status status);

  /*Determine status from parts*/
  // TODO(wlr): this should be private
  Task::Status determine_status(bool spawnable, bool mappable, bool reservable,
                                bool ready);

  /*Get the task status*/
  Task::Status get_status() const {
    const Task::Status status = this->status.load();
    return status;
  }

  /* Set complete */
  void set_complete();

  /* Get complete */
  bool get_complete();

  /// TODO(hc): move these to cpp.
  /// TODO(hc): Camel or snake case?
  void SetMappedDevice(Device *dev) {
    assert(mapped_device_ == NULL);
    this->mapped_device_ = dev;
  }

  const Device &GetMappedDevice() {
    assert(mapped_device_ != NULL);
    return *this->mapped_device_;
  }

private:
  Device *mapped_device_;
};

class InnerDataTask : public InnerTask {
public:
  InnerDataTask() : InnerTask() { this->has_data = true; }
};

#ifdef PARLA_ENABLE_LOGGING
LOG_ADAPT_STRUCT(InnerTask, name, instance, get_state)
LOG_ADAPT_DERIVED(InnerDataTask, (InnerTask))
#endif

/**
 *   The C++ "Mirror" of Parla's Python Workers
 *   This class is used to create a C++ representation of a Parla Worker
 *   All scheduling logic should be handled by these after creation until
 * launched by the Python callback
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

  /* Task Buffer (for enqueing new ready tasks at task cleanup ) */
  TaskStateList enqueue_buffer;

  // TODO: (improvement?) Custom Barrier and Event Handling

  // TODO: (improvement?) A buffer for multiple tasks assigned to a worker

  InnerWorker() = default;
  InnerWorker(void *worker) : py_worker(worker){};

  /* Set the Python Worker */
  void set_py_worker(void *worker) { this->py_worker = worker; };

  /*Set the scheduler*/
  void set_scheduler(InnerScheduler *scheduler) {
    this->scheduler = scheduler;
  };

  /* Set the thread idx */
  void set_thread_idx(int idx) { this->thread_idx = idx; };

  /* Wait for a task to be assigned */
  void wait();

  /* Assign a task to the worker and notify worker that it is available*/
  void assign_task(InnerTask *task);

  /* Get task */
  InnerTask *get_task();

  /* Remove task */
  void remove_task();

  void stop();
};

#ifdef PARLA_ENABLE_LOGGING
LOG_ADAPT_STRUCT(InnerWorker, thread_idx, notified)
#endif

template <typename AllWorkers_t, typename ActiveWorkers_t> class WorkerPool {

public:
  /* Container of all workers */
  AllWorkers_t all_workers;

  /* Container of available workers */
  ActiveWorkers_t active_workers;

  /* Number of workers */
  int max_workers;

  /*Mutex for blocking spawn/await*/
  std::mutex mtx;

  /*Condition variable for blocking spawn/await*/
  std::condition_variable cv;

  /* Number of notified but not running workers*/
  std::atomic<int> notified_workers{0};

  WorkerPool() = default;
  WorkerPool(int nworkers) : max_workers(nworkers){};

  /* Add a worker to the active pool */
  void enqueue_worker(InnerWorker *worker);

  /* Remove a worker from the active pool */
  InnerWorker *dequeue_worker();

  /* Add a worker to the all pool */
  void add_worker(InnerWorker *worker);

  /* Get number of available workers */
  int get_num_available_workers();

  /* Get number of total workers */
  int get_num_workers();

  /* Set number of total workers */
  void set_num_workers(int nworkers);

  /*Increase number of notified workers*/
  int increase_num_notified_workers();

  /*Decrease number of notified workers*/
  int decrease_num_notified_workers();

  /*Get number of notified workers*/
  inline int get_num_notified_workers() {
    return this->notified_workers.load();
  }

  /*Blocking for spawn/await so that other threads can take the GIL*/
  void spawn_wait();

  /* Remove a worker from the all pool */
  // void remove_worker(InnerWorker* worker);
};

typedef WorkerPool<WorkerQueue, WorkerQueue> WorkerPool_t;

// Forward declaration of scheduler phases

class Mapper;
class MemoryReserver;
class RuntimeReserver;
class Launcher;

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
  /* Sleep Between Loops */
  bool sleep_flag = false;

  /* Time to sleep between loops (microseconds) */
  int sleep_time = 20;

  /* Task Buffer */
  std::vector<InnerTask *> task_buffer = std::vector<InnerTask *>(10);

  /* Container of Thread Workers */
  WorkerPool_t workers;

  /* Resource Pool */
  InnerResourcePool<float>
      *resources; // TODO: Dummy class, needs complete rework with devices

  /* Active task counter (thread-safe) */
  std::atomic<int> num_active_tasks{1};

  /* Should Run, Stop Condition */
  std::atomic<bool> should_run = true;

  /* Phase: maps tasks to devices */
  Mapper *mapper;

  /* Phase reserves resources to limit/plan task execution*/
  MemoryReserver *memory_reserver;
  RuntimeReserver *runtime_reserver;

  /*Responsible for launching a task. Signals worker thread*/
  Launcher *launcher;

  InnerScheduler(DeviceManager *device_manager);
  // InnerScheduler(int nworkers);

  /* Pointer to callback to stop the Python scheduler */
  stopfunc_t stop_callback;

  /* Pointer to Python scheduler */
  void *py_scheduler;

  /* Scheduler Status */
  Scheduler::Status status;

  /* Set the number of workers */
  void set_num_workers(int nworkers);

  /* Set available resources  */
  void set_resources(std::string resource_name,
                     float resource_value); // TODO: Dummy function, needs
                                            // complete rework with devices

  /* Set Python Scheduler */
  void set_py_scheduler(void *py_scheduler);

  /* Set Python "stop" callback */
  void set_stop_callback(stopfunc_t stop_callback);

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
  void enqueue_task(InnerTask *task, Task::StatusFlags flags);

  /* Enqueue more than one task */
  void enqueue_tasks(TaskStateList &tasks);

  /* Add worker */
  void add_worker(InnerWorker *worker);

  /* Enqueue worker. */
  void enqueue_worker(InnerWorker *worker);

  /* Complete all task finalization. Notify Dependents / Release Resources /
   * Worker Enqueue */
  void task_cleanup(InnerWorker *worker, InnerTask *task, int state);

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

  /* Get number of noitified workers*/
  int get_num_notified_workers() {
    return this->workers.get_num_notified_workers();
  }

  /* Spawn wait. Slow down the compute bound spawning thread so tasks on other
   * threads can start*/
  void spawn_wait();

protected:
  DeviceManager *device_manager_;
};

#endif // PARLA_BACKEND_HPP
