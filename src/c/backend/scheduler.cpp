#include "include/phases.hpp"
#include "include/policy.hpp"
#include "include/resources.hpp"
#include "include/runtime.hpp"

#include <new>

#ifdef PARLA_ENABLE_LOGGING
namespace binlog {
int global_reset_count = 0;
}
#endif

// Worker Implementation

void InnerWorker::wait() {
  NVTX_RANGE("worker:wait", NVTX_COLOR_CYAN)
  LOG_INFO(WORKER, "Worker waiting: {}", this->thread_idx);
  std::unique_lock<std::mutex> lck(mtx);
  // std::cout << "Waiting for task (C++) " << this->thread_idx << std::endl;
  cv.wait(lck, [this] { return this->notified; });
  // std::cout << "Task assigned (C++) " << this->thread_idx << " " <<
  // this->ready
  //           << std::endl;
  // std::cout << "Task assigned (C++) " << this->task->get_name() << ": "
  //           << this->task->instance << std::endl;
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

void InnerWorker::get_task(InnerTask **task, bool *is_data_task) {
  this->scheduler->decrease_num_notified_workers();
  *task = this->task;
  *is_data_task = this->task->is_data_task();
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

InnerScheduler::InnerScheduler(DeviceManager *device_manager)
    : device_manager_(device_manager), parray_tracker_(device_manager) {

  // A dummy task count is used to keep the scheduler alive.
  // NOTE: At least one task must be added to the scheduler by the main thread,
  // otherwise the runtime will finish immediately
  // this->increase_num_active_tasks();

  this->workers.set_num_workers(1);

  // Mapping policy
  std::shared_ptr<LocalityLoadBalancingMappingPolicy> mapping_policy =
      std::make_shared<LocalityLoadBalancingMappingPolicy>(
          device_manager, &this->parray_tracker_);

  // Initialize the phases
  this->mapper = new Mapper(this, device_manager, std::move(mapping_policy));
  this->memory_reserver = new MemoryReserver(this, device_manager);
  this->runtime_reserver = new RuntimeReserver(this, device_manager);
  this->launcher = new Launcher(this, device_manager);
  // this->resources = std::make_shared < ResourcePool<std::atomic<int64_t>>();
}

InnerScheduler::~InnerScheduler() {
  delete this->mapper;
  delete this->memory_reserver;
  delete this->runtime_reserver;
  delete this->launcher;
}

void InnerScheduler::set_num_workers(int nworkers) {
  this->workers.set_num_workers(nworkers);
}

void InnerScheduler::set_py_scheduler(void *py_scheduler) {
  this->py_scheduler = py_scheduler;
}

void InnerScheduler::set_stop_callback(stopfunc_t stop_callback) {
  this->stop_callback = stop_callback;
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

  this->mapper->run(this->memory_reserver);
  this->memory_reserver->run(this->runtime_reserver);
  this->runtime_reserver->run(this->launcher);

  // LOG_TRACE(SCHEDULER, "ReadyPhase Status: {}", this->runtime_reserver);
  return this->status;
}

void InnerScheduler::activate_wrapper() { this->activate(); }

void InnerScheduler::spawn_task(InnerTask *task) {
  LOG_INFO(SCHEDULER, "Spawning task: {}", task);
  NVTX_RANGE("Scheduler::spawn_task", NVTX_COLOR_RED)

  auto status = task->process_dependencies();
  this->increase_num_active_tasks();
  task->set_state(TaskState::SPAWNED);
  this->enqueue_task(task, status);
}

void InnerScheduler::enqueue_task(InnerTask *task, TaskStatusFlags status) {
  // TODO: Change this to appropriate phase as it becomes implemented
  LOG_INFO(SCHEDULER, "Enqueing task: {}, Status: {}", task, status);
  // std::cout << "Enqueing task: " << task->get_name()
  //           << " Instance: " << task->instance << std::endl;
  if (status.mappable && (task->get_state() < TaskState::MAPPED)) {
    LOG_INFO(SCHEDULER, "Enqueing task: {} to mapper", task);
    task->set_status(TaskStatus::MAPPABLE);
    this->mapper->enqueue(task);
  } else if (status.reservable && (task->get_state() == TaskState::MAPPED)) {
    task->set_status(TaskStatus::RESERVABLE);
    LOG_INFO(SCHEDULER, "Enqueing task: {} to memory reserver", task);
    this->memory_reserver->enqueue(task);
  } else if (status.runnable && (task->get_state() == TaskState::RESERVED)) {
    task->set_status(TaskStatus::RUNNABLE);
    // std::cout << "ENQUEUE FROM CALLBACK" << std::endl;
    LOG_INFO(SCHEDULER, "Enqueing task: {} to runtime reserver", task);
    this->runtime_reserver->enqueue(task);
  }
}

void InnerScheduler::enqueue_tasks(TaskStatusList &tasks) {
  // LOG_INFO(SCHEDULER, "Enqueing tasks: {}", tasks);
  for (TaskStatusPair task_status : tasks) {
    this->enqueue_task(task_status.first, task_status.second);
  }
}

void InnerScheduler::add_worker(InnerWorker *worker) {
  LOG_INFO(SCHEDULER, "Adding worker {} to pool", worker);
  this->workers.add_worker(worker);
}

void InnerScheduler::enqueue_worker(InnerWorker *worker) {
  LOG_INFO(SCHEDULER, "Enqueuing worker: {} is ready.", worker);
  this->workers.enqueue_worker(worker);
}

void InnerScheduler::task_cleanup_presync(InnerWorker *worker, InnerTask *task,
                                          int state_int) {
  NVTX_RANGE("Scheduler::task_cleanup_presync", NVTX_COLOR_MAGENTA)
  LOG_INFO(WORKER, "Cleaning up: {} on  {}", task, worker);

  TaskState state = static_cast<TaskState>(state_int);

  // std::cout << "CLEANUP PRE SYNC: " << state << " " << Task::RUNAHEAD
  //           << std::endl;

  // std::cout << "Task state: " << state << std::endl;
  if (state == TaskState::RUNAHEAD) {

    // Notify dependents that they can be scheduled
    auto &enqueue_buffer = worker->enqueue_buffer;
    task->notify_dependents(enqueue_buffer, TaskState::RUNAHEAD);
    if (enqueue_buffer.size() > 0) {
      this->enqueue_tasks(enqueue_buffer);
      enqueue_buffer.clear();
    }
  }
}

void InnerScheduler::task_cleanup_postsync(InnerWorker *worker, InnerTask *task,
                                           int state_int) {
  NVTX_RANGE("Scheduler::task_cleanup_postsync", NVTX_COLOR_MAGENTA)

  TaskState state = static_cast<TaskState>(state_int);

  // std::cout << "Task Cleanup Post Sync" << std::endl;

  if (state == TaskState::RUNAHEAD) {
    this->decrease_num_active_tasks();
    task->notify_dependents_completed();
  }

  // Release all resources for this task on all devices
  for (size_t i = 0; i < task->assigned_devices.size(); ++i) {
    Device *device = task->assigned_devices[i];
    DevID_t dev_id = device->get_global_id();
    ResourcePool_t &device_pool = device->get_reserved_pool();
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];

    // TODO(wlr): This needs to be changed to not release PARRAY resources
    device_pool.increase<ResourceCategory::All>(task_pool);

    // PArrays could be evicted even during task barrier continuation.
    // However, these PArrays will be allocated and tracked
    // again after the task restarts.
    if (!task->is_data_task()) {
      for (size_t j = 0; j < task->parray_list[i].size(); ++j) {
        parray::InnerPArray *parray = task->parray_list[i][j].first;
        parray->decr_num_active_tasks(dev_id);
        // This PArray is not released from the PArray tracker here,
        // but when it is EVICTED, it will check the number of referneces
        // and will be released if that is 0.
      }
    }
    this->mapper->atomic_decr_num_mapped_tasks_device(dev_id);
  }

  // Clear all assigned streams from the task
  task->streams.clear();
  this->launcher->num_running_tasks--;
  worker->remove_task();

  if (state == TaskState::RUNNING) {
    task->reset();
    auto status = task->process_dependencies();
    this->enqueue_task(task, status);
  }

  this->enqueue_worker(worker);
}

void InnerScheduler::task_cleanup(InnerWorker *worker, InnerTask *task,
                                  int state) {
  NVTX_RANGE("Scheduler::task_cleanup", NVTX_COLOR_MAGENTA)

  task_cleanup_presync(worker, task, state);
  // synchronize task enviornment
  task->synchronize_events();
  task_cleanup_postsync(worker, task, state);
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
  return this->launcher->get_count();
}

int InnerScheduler::get_num_ready_tasks() {
  return this->runtime_reserver->get_count();
}

void InnerScheduler::spawn_wait() { this->workers.spawn_wait(); }
