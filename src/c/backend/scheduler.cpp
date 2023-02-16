#include "include/phases.hpp"
#include "include/runtime.hpp"

// Worker Implementation

void InnerWorker::wait() {
  NVTX_RANGE("worker:wait", NVTX_COLOR_CYAN)
  LOG_INFO(WORKER, "Worker waiting: {}", this->thread_idx);
  std::unique_lock<std::mutex> lck(mtx);
  // std::cout << "Waiting for task (C++) " << this->thread_idx << std::endl;
  cv.wait(lck, [this] { return this->notified; });
  // std::cout << "Task assigned (C++) " << this->thread_idx << " "
  //           << this->ready << std::endl;
  this->scheduler->increase_num_notified_workers();
}

void InnerWorker::assign_task(InnerTask *task) {
  NVTX_RANGE("worker:assign_task", NVTX_COLOR_CYAN)
  // std::cout << "Assigning task (C++) " << this->thread_idx << " "
  //           << this->ready << std::endl;
  assert(ready == false);
  std::unique_lock<std::mutex> lck(mtx);
  this->task = task;
  this->ready = true;
  this->notified = true;
  cv.notify_one();
}

InnerTask *InnerWorker::get_task() {
  this->scheduler->decrease_num_notified_workers();
  return this->task;
}

void InnerWorker::remove_task() {
  // std::cout << "Removing task (C++) " << this->thread_idx << " "
  //           << this->task->name << std::endl;
  std::unique_lock<std::mutex> lck(mtx);
  this->task = nullptr;
  this->ready = false;
  this->notified = false;
}

void InnerWorker::stop() {
  // signal cv so that we can terminate
  std::unique_lock<std::mutex> lck(mtx);
  LOG_INFO(WORKER, "Worker stopping: {}", this->thread_idx);
  this->notified = true;
  cv.notify_all();
}
// WorkerPool Implementation

template <typename AllWorkers_t, typename ActiveWorkers_t>
void WorkerPool<AllWorkers_t, ActiveWorkers_t>::enqueue_worker(
    InnerWorker *worker) {
  this->active_workers.push_back(worker);
  // std::cout << "Enqueued Worker ID: " << worker->thread_idx << std::endl;
  // std::cout << "Active workers: " << this->active_workers.atomic_size()
  //           << std::endl;
}

template <typename AllWorkers_t, typename ActiveWorkers_t>
InnerWorker *WorkerPool<AllWorkers_t, ActiveWorkers_t>::dequeue_worker() {
  InnerWorker *worker = this->active_workers.back_and_pop();
  // std::cout << "Dequeued Worker ID: " << worker->thread_idx << std::endl;
  return worker;
}

template <typename AllWorkers_t, typename ActiveWorkers_t>
void WorkerPool<AllWorkers_t, ActiveWorkers_t>::add_worker(
    InnerWorker *worker) {
  // std::cout << "Adding worker: " << worker->thread_idx << std::endl;
  this->all_workers.push_back(worker);
  assert(this->all_workers.size() <= this->max_workers);
}

template <typename AllWorkers_t, typename ActiveWorkers_t>
int WorkerPool<AllWorkers_t, ActiveWorkers_t>::get_num_available_workers() {
  auto num_workers = this->active_workers.atomic_size();
  // std::cout << "Available workers: " << num_workers << std::endl;
  return num_workers;
}

template <typename AllWorkers_t, typename ActiveWorkers_t>
int WorkerPool<AllWorkers_t, ActiveWorkers_t>::get_num_workers() {
  return this->max_workers;
}

template <typename AllWorkers_t, typename ActiveWorkers_t>
void WorkerPool<AllWorkers_t, ActiveWorkers_t>::set_num_workers(int nworkers) {
  this->max_workers = nworkers;
}

template <typename AllWorkers_t, typename ActiveWorkers_t>
int WorkerPool<AllWorkers_t, ActiveWorkers_t>::increase_num_notified_workers() {
  int before = this->notified_workers.fetch_add(1);
  return before;
}

template <typename AllWorkers_t, typename ActiveWorkers_t>
int WorkerPool<AllWorkers_t, ActiveWorkers_t>::decrease_num_notified_workers() {
  int before = this->notified_workers.fetch_sub(1);
  return before;
}

template <typename AllWorkers_t, typename ActiveWorkers_t>
void WorkerPool<AllWorkers_t, ActiveWorkers_t>::spawn_wait() {

  std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
}

template class WorkerPool<WorkerQueue, WorkerQueue>;

// Scheduler Implementation

InnerScheduler::InnerScheduler() {

  // A dummy task count is used to keep the scheduler alive.
  // NOTE: At least one task must be added to the scheduler by the main thread,
  // otherwise the runtime will finish immediately
  // this->increase_num_active_tasks();

  this->workers.set_num_workers(1);

  // Initialize the phases
  this->ready_phase = new ReadyPhase(this);
  this->launcher = new LauncherPhase(this);
  this->resources = new InnerResourcePool<float>();
  // TODO: Clean these up
}

void InnerScheduler::set_num_workers(int nworkers) {
  this->workers.set_num_workers(nworkers);
}

void InnerScheduler::set_resources(std::string resource_name,
                                   float resource_value) {
  this->resources->set(resource_name, resource_value);
}

void InnerScheduler::set_py_scheduler(void *py_scheduler) {
  this->py_scheduler = py_scheduler;
}

void InnerScheduler::set_stop_callback(stopfunc_t stop_callback) {
  this->stop_callback = stop_callback;
}

void InnerScheduler::set_launch_callback(launchfunc_t launch_callback) {
  this->launcher->set_launch_callback(launch_callback, this->py_scheduler);
}

void InnerScheduler::run() {
  NVTX_RANGE("Scheduler::run", NVTX_COLOR_RED)
  unsigned long long iteration_count = 0;
  while (this->should_run.load()) {
    auto status = this->activate();
    if (this->sleep_flag) {
      std::this_thread::sleep_for(std::chrono::milliseconds(this->sleep_time));
    }
  }
}

void InnerScheduler::stop() {
  LOG_INFO(SCHEDULER, "Stopping scheduler");
  this->should_run = false;
  launch_stop_callback(this->stop_callback, this->py_scheduler);
  LOG_INFO(SCHEDULER, "Stopped scheduler");
}

Scheduler::Status InnerScheduler::activate() {
  // std::cout<< "Scheduler Activated" << std::endl;
  // this->spawned_phase->run(this->mapped_phase);
  // this->mapped_phase->run(this->reserved_phase);
  // this->reserved_phase->run(this->ready_phase);
  this->ready_phase->run(this->launcher);
  // this->launcher->run();
  // this->ready_phase->status.print();
  // LOG_TRACE(SCHEDULER, "ReadyPhase Status: {}", this->ready_phase);
  return this->status;
}

void InnerScheduler::activate_wrapper() { this->activate(); }

void InnerScheduler::enqueue_task(InnerTask *task) {
  // TODO: Change this to appropriate phase as it becomes implemented
  LOG_INFO(SCHEDULER, "Enqueing task: {}", task);
  this->ready_phase->enqueue(task);
}

void InnerScheduler::enqueue_tasks(std::vector<InnerTask *> &tasks) {
  LOG_INFO(SCHEDULER, "Enqueing tasks: {}", tasks);
  this->ready_phase->enqueue(tasks);
}

void InnerScheduler::add_worker(InnerWorker *worker) {
  LOG_INFO(SCHEDULER, "Adding worker {} to pool", worker);
  this->workers.add_worker(worker);
}

void InnerScheduler::enqueue_worker(InnerWorker *worker) {
  LOG_INFO(SCHEDULER, "Enqueuing worker: {} is ready.", worker);
  this->workers.enqueue_worker(worker);
}

void InnerScheduler::task_cleanup(InnerWorker *worker, InnerTask *task,
                                  int state) {
  NVTX_RANGE("Scheduler::task_cleanup", NVTX_COLOR_MAGENTA)
  LOG_INFO(WORKER, "Cleaning up: {} on  {}", task, worker);

  // std::cout << "Task Cleanup: " << task->name << " " << state << std::endl;

  /* Task::States are: spawned, mapped, reserved, ready, running, complete */

  // This will be called by EVERY thread that finishes a task
  // Everything in here needs to be thread safe

  // TODO: for runahead, we need to do this AFTER the task body is complete
  //      Need to add back to the pool after notify_dependents
  //      Movin this below but leaving my original placement here for now
  // this->resources->increase(task->resources);
  // this->launcher->num_running_tasks--;
  // this->workers.enqueue_worker(worker);

  // std::cout << "Task Cleanup: " << task->name << " " << state << std::endl;

  this->launcher->num_running_tasks--;

  if (state == Task::running) {
    // std::cout << "Task Continuation (C++) " << task->name << " " << state
    //          << std::endl;
    // Do continuation handling
    // TODO:
    //  - make sure state ids match
    //  - add and process dependencies
    //  - if true, enqueue task
    bool status = task->process_dependencies();

    if (status) {
      this->enqueue_task(task);
    }
  }

  if (state == Task::complete) {
    // When a task completes we need to notify all of its dependents
    // and enqueue them if they are ready

    // std::cout << "Task Complete: " << task->name << std::endl;

    auto &enqueue_buffer = worker->enqueue_buffer;
    task->notify_dependents(enqueue_buffer);
    if (enqueue_buffer.size() > 0) {
      this->enqueue_tasks(enqueue_buffer);
      enqueue_buffer.clear();
    }

    // TODO: Wait on CUDA events here for runahead
    // TODO: Should probably split this into two functions here
    //      Then they can be called separately in Python

    // We also need to decrease the number of active tasks
    // If this is the last active task, the scheduler is stopped
    this->decrease_num_active_tasks();
  }

  // TODO: for runahead, we need to do this AFTER the task body is complete
  //      Need to add back to the pool after notify_dependents
  worker->remove_task();
  this->resources->increase(task->resources);
  this->enqueue_worker(worker);

  // NOTE: Task::complete is NOT equivalent to task->complete
  // Task->complete:
  //      - only signals that notify_dependencies has been completed
  //      - Has a valid state for use in dependency handling

  // Task::complete:
  //     - signals that the task has finished everything

  task->set_state(state);
}

int InnerScheduler::get_num_active_tasks() { return this->num_active_tasks; }

void InnerScheduler::increase_num_active_tasks() {
  int count = this->num_active_tasks.fetch_add(1);
  // std::cout << "Increasing num active tasks: " << count + 1 << std::endl;
}

void InnerScheduler::decrease_num_active_tasks() {

  int count = this->num_active_tasks.fetch_sub(1) - 1;

  // std::cout << "Decreasing num active tasks: " << count << std::endl;

  if (count == 0) {
    this->stop();
  }
}

int InnerScheduler::increase_num_notified_workers() {
  return this->workers.increase_num_notified_workers();
}

int InnerScheduler::decrease_num_notified_workers() {
  return this->workers.decrease_num_notified_workers();
}

int InnerScheduler::get_num_running_tasks() {
  return this->launcher->num_running_tasks;
}

int InnerScheduler::get_num_ready_tasks() {
  return this->ready_phase->get_count();
}

void InnerScheduler::spawn_wait() { this->workers.spawn_wait(); }
