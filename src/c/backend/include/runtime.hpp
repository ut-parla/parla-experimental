#pragma once
#include <condition_variable>
#ifndef PARLA_BACKEND_HPP

#include <atomic>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>

using namespace std::chrono_literals;

#include "containers.hpp"

#include <nvtx3/nvtx3.hpp>
struct my_domain {
  static constexpr char const *name{"Parla Runtime"};
};

struct compute_domain {
  static constexpr char const *name{"Compute"};
};

using my_scoped_range = nvtx3::scoped_range_in<my_domain>;
using compute_range = nvtx3::scoped_range_in<compute_domain>;

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
  compute_range r("sleep::busy", nvtx3::rgb{0, 127, 127});
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

/*State of the task. Shows which phase it is in.*/
enum State { created, spawned, mapped, reserved, ready, running, complete };
} // namespace Task

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

  /* Status of the task */
  std::atomic<Task::State> state{Task::created};

  /*Task monitor*/
  std::mutex mtx;

  /* Whether the task has finished running. */
  std::atomic<bool> complete{false};

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

  /* Number of blocking (uncompleted) dependencies */
  std::atomic<int> num_blocking_dependencies{0};

  /* Number of mapped dependencies */
  std::atomic<int> num_mapped_dependencies{0};

  /* Number of reserved dependencies */
  std::atomic<int> num_reserved_dependencies{0};

  /* Tasks Internal Resource Pool. */
  InnerResourcePool<float> resources;

  InnerTask();
  InnerTask(long long int id, void *py_task);
  InnerTask(std::string name, long long int id, void *py_task);

  /* Set the name of the task */
  void set_name(std::string name);

  /* Set the id of the task */
  void set_id(long long int name);

  /* Set the python task */
  void set_py_task(void *py_task);

  /* Set the priority of the task */
  void set_priority(int priority);

  /*Set a resource of the task*/
  void set_resources(std::string resource_name, float resource_value);

  // template<typename T>
  // void set_resources(InnerResourcePool<T> resource_pool);

  /*Get a resource of the task*/
  // template<typename T>
  // T get_resources(std::string resource_name);

  /* Add a dependency to the task buffer but don't process it*/
  void queue_dependency(InnerTask *task);

  /* Add a list of dependencies to the task. For external use.*/
  bool process_dependencies();

  /* Clear the dependency list */
  void clear_dependencies();

  /* Add a dependency to the task and process it*/
  bool add_dependency(InnerTask *task);

  /* Add a list of dependencies to the task and process them. For external
   * use.*/
  bool add_dependencies(std::vector<InnerTask *> &tasks);

  /* Add a dependent to the task */
  bool add_dependent(InnerTask *task);

  /* Add a list of dependents to the task */
  // void add_dependents(std::vector<bool> result, std::vector<InnerTask*>&
  // tasks);

  /*
   *  Notify dependents that dependencies have completed
   *  This should be called by the worker when a task has completed
   *  Returns a container of tasks that are now ready to run
   *  TODO: Decide on a container to use for this
   */
  std::vector<InnerTask *> &notify_dependents(std::vector<InnerTask *> &tasks);

  /* Wrapper for testing */
  bool notify_dependents_wrapper();

  /* Notify the task that one of its dependents has completed
   *  Decrements the number of blocking dependencies.
   *  Return true if 0 blocking dependencies remain.
   *  Used by "notify_dependents"
   */
  bool notify();

  /* Return whether the task is ready to run */
  bool blocked();

  /* Get number of dependencies */
  int get_num_dependencies();

  /* Get number of dependents */
  int get_num_dependents();

  /* Get number of blocking dependencies */
  int get_num_blocking_dependencies();

  /* Get dependency list. Used for testing Python interface. */
  std::vector<void *> get_dependencies();

  /* Get dependents list. Used for testing Python interface. */
  std::vector<void *> get_dependents();

  /* Get python task */
  void *get_py_task();

  /* Set the task status */
  void set_state(int state);

  /* Set the task status */
  void set_state(Task::State state);

  /* Set complete */
  void set_complete(bool complete);

  /* Get complete */
  bool get_complete();
};

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

  std::mutex mtx;
  std::condition_variable cv;
  bool ready = false;
  bool notified = false;

  int thread_idx = -1;

  /* Task Buffer (for enqueing new ready tasks at task cleanup ) */
  std::vector<InnerTask *> enqueue_buffer;

  // TODO: (improvement?) Custom Barrier and Event Handling

  // TODO: (improvement?) A buffer for multiple tasks assigned to a worker

  InnerWorker() = default;
  InnerWorker(void *worker) : py_worker(worker){};

  /* Set the Python Worker */
  void set_py_worker(void *worker) { this->py_worker = worker; };

  /* Set the thread idx */
  void set_thread_idx(int idx) { this->thread_idx = idx; };

  /* Wait for a task to be assigned */
  void wait() {
    my_scoped_range r("worker::wait", nvtx3::rgb{127, 127, 0});
    std::unique_lock<std::mutex> lck(mtx);
    // std::cout << "Waiting for task (C++) " << this->thread_idx << std::endl;
    cv.wait(lck, [this] { return this->notified; });
    // cv.wait(lck);
    // std::cout << "Task assigned (C++) " << this->thread_idx << " " <<
    // this->ready << std::endl;
  };

  /* Assign a task to the worker and notify worker that it is available*/
  void assign_task(InnerTask *task) {
    my_scoped_range r("worker:assign_task", nvtx3::rgb{127, 127, 0});
    // std::cout << "Assigning task (C++) " << this->thread_idx << " " <<
    // this->ready << std::endl;
    assert(ready == false);
    std::unique_lock<std::mutex> lck(mtx);
    this->task = task;
    this->ready = true;
    this->notified = true;
    cv.notify_one();
  };

  /* Remove task */
  void remove_task() {
    // std::cout << "Removing task (C++) " << this->thread_idx << " " <<
    // this->task->name << std::endl;
    std::unique_lock<std::mutex> lck(mtx);
    this->task = nullptr;
    this->ready = false;
    this->notified = false;
  };

  void stop() {
    // signal cv so that we can terminate
    std::unique_lock<std::mutex> lck(mtx);
    // std::cout << "Stopping worker (C++) " << this->thread_idx << std::endl;
    this->notified = true;
    cv.notify_all();
  }
};

template <typename AllWorkers_t, typename ActiveWorkers_t> class WorkerPool {

public:
  /* Container of all workers */
  AllWorkers_t all_workers;

  /* Container of available workers */
  ActiveWorkers_t active_workers;

  /* Number of workers */
  int max_workers;

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

  /* Remove a worker from the all pool */
  // void remove_worker(InnerWorker* worker);
};

typedef WorkerPool<WorkerQueue, WorkerQueue> WorkerPool_t;

// Forward declaration of scheduler phases

class SpawnedPhase;
class MappedPhase;
class ReservedPhase;
class ReadyPhase;
class LauncherPhase;

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

  /* Incapsulation of Scheduler Ready Phase */
  ReadyPhase *ready_phase;

  /*Responsible for launching a task. Holds python launch callback*/
  LauncherPhase *launcher;

  InnerScheduler();
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

  /* Set Python "launch" callback */
  void set_launch_callback(launchfunc_t launch_callback);

  /* Run the scheduler thread. Active for the lifetime of the Parla program */
  void run();

  /*Stop the scheduler. Called at the end of the Parla program */
  void stop();

  /* Activate scheduler on current thread. Runs through scheduler phases. */
  Scheduler::Status activate();

  /* Activate wrapper for Python layer (for use as scheduler callback) */
  void activate_wrapper();

  /* Enqueue task. */
  void enqueue_task(InnerTask *task);

  /* Enqueue more than one task */
  void enqueue_tasks(std::vector<InnerTask *> &tasks);

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

  /* Get number of running tasks. A task is running if is Python task has been
   * assigned and the task is not complete*/
  int get_num_running_tasks();

  /* Get number of ready tasks. A task is ready if its dependencies has been
   * dispatched to a hardware queue or are complete */
  int get_num_ready_tasks();
};

#endif // PARLA_BACKEND_HPP