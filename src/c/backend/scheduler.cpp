#include "include/device.hpp"
#include "include/parray.hpp"
#include "include/phases.hpp"
#include "include/policy.hpp"
#include "include/resources.hpp"
#include "include/runtime.hpp"

#include <cstdint>
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
  // std::cout << "Task assigned (C++) " << this->thread_idx << " "
  //           << this->ready << std::endl;
  this->scheduler->increase_num_notified_workers();
}

void InnerWorker::assign_task(InnerTask *task) {
  NVTX_RANGE("worker:assign_task", NVTX_COLOR_CYAN)
  //std::cout << "Assigning task (C++) " << this->thread_idx << " "
  //          << this->ready << ", " << task->name << std::endl;
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

void InnerWorker::record_task_begin_epochs() {
  if (!this->task->is_data_task()) {
    TimePoint initial_time_epoch = this->scheduler->get_initial_epoch();
    TimePoint now = std::chrono::system_clock::now();
    double task_begin_epochs = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - initial_time_epoch).count();
    //std::cout << this->task->name << "'s completion time:" << task_completion_epochs << "\n";
    this->task->record_task_begin_epochs(task_begin_epochs);
  }
}

void InnerWorker::record_task_completion_epochs() {
  // Only consider computation tasks.
  if (!this->task->is_data_task()) {
    TimePoint initial_time_epoch = this->scheduler->get_initial_epoch();
    TimePoint now = std::chrono::system_clock::now();
    double task_completion_epochs = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - initial_time_epoch).count();
    //std::cout << this->task->name << "'s completion time:" << task_completion_epochs << "\n";
    this->task->record_task_completion_epochs(task_completion_epochs);
    log_rl_msg(2, "device selection,"+task->name+", "+
        std::to_string(this->task->assigned_devices[0]->get_global_id())+", "+
        std::to_string(this->task->begin_time_epochs)+", "+
        std::to_string(this->task->completion_time_epochs)+", "+
        std::to_string(this->task->max_depcompl_time_epochs)+", "+
        std::to_string(this->task->completion_time_epochs -
          this->task->max_depcompl_time_epochs));

  }
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

<<<<<<< HEAD
InnerScheduler::InnerScheduler(LRUGlobalEvictionManager* memory_manager,
    DeviceManager *device_manager, MappingPolicyType mapping_policy)
    : mm_(memory_manager), device_manager_(device_manager),
    parray_tracker_(device_manager) {
  // For now, it does not evict PArrays on CPU memory.
  this->memory_size_to_evict.resize(
      device_manager->template get_num_devices<DeviceType::All>());

  // A dummy task count is used to keep the scheduler alive.
  // NOTE: At least one task must be added to the scheduler by the main thread,
  // otherwise the runtime will finish immediately
  // this->increase_num_active_tasks();

  this->workers.set_num_workers(1);

  DevID_t num_devices = device_manager_->get_num_devices<ParlaDeviceType::All>();
  this->dev_num_mapped_tasks_.resize(num_devices);
  this->dev_num_tasks_mapped_states_.resize(num_devices);
  this->dev_num_tasks_resreserved_states_.resize(num_devices);
  this->dev_num_ready_tasks_.resize(num_devices);
  this->dev_num_running_tasks_.resize(num_devices);
  // Initialize the phases
  this->mapper = new Mapper(this, device_manager, mapping_policy);
  this->memory_reserver = new MemoryReserver(this, device_manager);
  this->runtime_reserver = new RuntimeReserver(this, device_manager);
  this->launcher = new Launcher(this, device_manager);
}

InnerScheduler::~InnerScheduler() {
  std::cout << " Number of eviction invocation:" <<
      this->num_eviction_invocation << "\n";
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

bool InnerScheduler::get_should_run() {
  return this->should_run.load();
}

void InnerScheduler::set_memory_size_to_evict(
    size_t size, DevID_t dev_id) {
  this->memory_size_to_evict[dev_id] = size;
}

size_t InnerScheduler::get_memory_size_to_evict(DevID_t dev_id) {
  return this->memory_size_to_evict[dev_id];
}

void InnerScheduler::run() {
  NVTX_RANGE("Scheduler::run", NVTX_COLOR_RED)
  while (this->should_run.load()) {
    this->break_for_eviction = false;
    auto status = this->activate();
    if (this->sleep_flag) {
      std::this_thread::sleep_for(std::chrono::milliseconds(this->sleep_time));
    }
    if (this->break_for_eviction) {
      ++this->num_eviction_invocation;
      // Yield a control to a Python scheduler to evict PArrays since
      // PArray coherency protocol is managed at there.
      break;
    }
  }
}

void InnerScheduler::stop() {
  LOG_INFO(SCHEDULER, "Stopping scheduler");
  this->should_run = false;
  this->mm_->print_stats();
  // XXX(hc): To process PArray eviction on Python,
  // Python scheduler now has an while loop that iterates until there is
  // no more task, and it wraps C scheduler's loop.
  // Therefore, there is no point for C++ scheduler to explicitly invoke
  // this callback at here. Python scheduler knows when it needs to stop.
  //launch_stop_callback(this->stop_callback, this->py_scheduler);
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
    //std::cout << "ENQUEUE FROM CALLBACK" << std::endl;
    LOG_INFO(SCHEDULER, "Enqueing task: {} to runtime reserver", task);
    this->runtime_reserver->enqueue(task);
  }

  this->task_name_to_task.emplace(task->name, task);
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

void InnerScheduler::create_parray(InnerPArray *parray, int parray_device_id) {

  PArrayAccess_t parray_access = std::make_pair(parray, AccessMode::NEW);

  auto device_manager = this->device_manager_;

  PArrayTracker *mapped_tracker = this->mapper->get_parray_tracker();
  PArrayTracker *reserved_tracker = this->memory_reserver->get_parray_tracker();

  DevID_t global_dev_id = parrayid_to_globalid(parray_device_id);

  size_t to_map = mapped_tracker->do_log(global_dev_id, parray_access);
  size_t to_reserve = reserved_tracker->do_log(global_dev_id, parray_access);
  add_unmapped_created_parray(parray, global_dev_id, to_map);
  add_unreserved_created_parray(parray, global_dev_id, to_reserve);
}

void InnerScheduler::remove_parray(InnerPArray *parray, DevID_t global_dev_id) {
  Device *device =
      this->device_manager_->get_device_by_global_id(global_dev_id);

  PArrayTracker *mapped_tracker = this->mapper->get_parray_tracker();
  PArrayTracker *reserved_tracker = this->memory_reserver->get_parray_tracker();

  // TODO: Decide policy for removal.
  // Simplest is just to call remove_parray on both trackers
  // Could also be to call DELETED or REMOVED status on do_log
}

void InnerScheduler::remove_parray_from_tracker(
    parray::InnerPArray *parray, DevID_t global_dev_id) {
  AccessMode access_mode = AccessMode::FREED;
  this->mapper->get_parray_tracker()->do_log(global_dev_id,
      std::make_pair(parray, access_mode));
  this->memory_reserver->get_parray_tracker()->do_log(global_dev_id,
      std::make_pair(parray, access_mode));
}

size_t InnerScheduler::get_mapped_memory(DevID_t global_dev_idx) {
  Device *device =
      this->device_manager_->get_device_by_global_id(global_dev_idx);
  auto &mapped_memory_pool = device->get_mapped_pool();
  return device->query_mapped<Resource::Memory>();
}

size_t InnerScheduler::get_reserved_memory(DevID_t global_dev_idx) {
  Device *device =
      this->device_manager_->get_device_by_global_id(global_dev_idx);
  return device->query_reserved<Resource::Memory>();
}

size_t InnerScheduler::get_max_memory(DevID_t global_dev_idx) {
  Device *device =
      this->device_manager_->get_device_by_global_id(global_dev_idx);
  return device->query_max<Resource::Memory>();
}

bool InnerScheduler::get_mapped_parray_state(DevID_t global_dev_idx,
                                             uint64_t parray_parent_id) {

  PArrayTracker *mapped_tracker = this->mapper->get_parray_tracker();
  return mapped_tracker->get_parray_state(global_dev_idx, parray_parent_id);
}

bool InnerScheduler::get_reserved_parray_state(DevID_t global_dev_idx,
                                               uint64_t parray_parent_id) {

  PArrayTracker *mapped_tracker = this->mapper->get_parray_tracker();
  return mapped_tracker->get_parray_state(global_dev_idx, parray_parent_id);
}

void InnerScheduler::task_cleanup_postsync(InnerWorker *worker, InnerTask *task,
                                           int state_int) {
  NVTX_RANGE("Scheduler::task_cleanup_postsync", NVTX_COLOR_MAGENTA)

  TaskState state = static_cast<TaskState>(state_int);

  // std::cout << "Task Cleanup Post Sync" << std::endl;
  DeviceManager *device_manager = this->device_manager_;

  if (state == TaskState::RUNAHEAD) {
    this->decrease_num_active_tasks();
    task->notify_dependents_completed();
  }

  // Release all resources for this task on all devices
  for (size_t local_device_idx = 0;
       local_device_idx < task->assigned_devices.size(); ++local_device_idx) {

    ParlaDevice *device = task->assigned_devices[local_device_idx];
    DevID_t dev_id = device->get_global_id();

    auto &device_reserved_pool = device->get_reserved_pool();
    auto &device_mapped_pool = device->get_mapped_pool();

    auto &task_pool = task->device_constraints[device->get_global_id()];

    // TODO(hc):This assumes that VCU is 1.
    device->begin_device_idle();

    device_reserved_pool.increase(task_pool);
    device_mapped_pool.decrease(task_pool);

    auto &parray_access_list = task->parray_list[local_device_idx];

    // PArrays could be evicted even during task barrier continuation.
    // However, these PArrays will be allocated and tracked
    // again after the task restarts.

    if (!task->is_data_task()) {
      for (size_t j = 0; j < parray_access_list.size(); ++j) {
        auto &parray_access = parray_access_list[j];
        InnerPArray *parray = parray_access.first;
        parray->decr_num_referring_tasks(dev_id);
        // Decrease this PArray's reference count.
        // If this becomes 0, this instance will be release
        // when the PArray coherency protocol updates it
        // to eviction state.
        this->release_parray_reference(parray, dev_id);
      }
      this->atomic_decr_num_mapped_tasks_device(dev_id);
      this->atomic_decr_num_running_tasks(dev_id);

      LOG_INFO("Debug",
          "Completed {} (Dev. {}): mapped {} res-reserved {} ready {} running {} total {}",
          task->name, dev_id,
          this->atomic_load_dev_num_tasks_mapped_states(dev_id),
          this->atomic_load_dev_num_tasks_resreserved_states(dev_id),
          this->atomic_load_dev_num_ready_tasks(dev_id),
          this->atomic_load_dev_num_running_tasks(dev_id),
          this->atomic_load_dev_num_mapped_tasks_device(dev_id));
      device->reduce_mapped_task_info(task->remote_data_bytes, task->num_dependencies,
          task->num_dependents);
    }

    if (task->is_data_task()) {
      // Decrease the number of mapped data tasks on the device
      // TODO(@dialecticDolt) Add this
      this->mapper->atomic_decr_num_mapped_data_tasks_device(dev_id);
    } else {
      // Decrease the number of mapped compute tasks on the device
      this->mapper->atomic_decr_num_mapped_tasks_device(dev_id);
    }
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
  if (task->name.find("begin_rl_task") == std::string::npos) {
    for (size_t i = 0; i < task->assigned_devices.size(); ++i) {
      ParlaDevice *device = task->assigned_devices[i];
      // TODO(hc):This assumes that VCU is 1.
      device->begin_device_idle();
    }
    TimePoint now = std::chrono::system_clock::now();
    TimePoint initial_time_epoch = this->get_initial_epoch();
    this->epoch_begin_epochs =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - initial_time_epoch).count();
  }
  task_cleanup_postsync(worker, task, state);
}

void InnerScheduler::complete_task_order_logs(InnerTask* task) {
  // after this update from end_task_graph, the next iteration would wake up.
  // so, the correct ordering is guaranteed.
  if (!this->is_task_mapping_log_registered()) {
    // The first iteration might not omit some tasks
    // that run after the first iteration.
    // e.g., Cholesky's Reset tasks
    this->task_log_register_counter.fetch_add(1);
    if (this->task_log_register_counter.load() == 2) {
      this->complete_task_mapping_log();
    }
  }
  if (!this->is_task_launching_log_registered()) {
    // TODO(hc): change this counter name
    if (this->task_log_register_counter.load() == 2) {
      this->complete_task_launching_log();
    }
  }
}

bool InnerScheduler::check_task_mapping_order_mode(uint32_t term = 50) {
  bool allow_proceeding{false};
  if (this->task_mapping_log_mode == 0) {
    allow_proceeding = false;
  } else if (this->task_mapping_log_mode == 1 ||
      this->task_mapping_log_mode == 2) {
    if (this->num_epochs == 1 || (this->num_epochs % (1 + term) == 0)) {
      // Allow log registration and its completion
      allow_proceeding = true;
    } else {
      allow_proceeding = false;
    }
  }
}

bool InnerScheduler::check_task_launching_order_mode(uint32_t term = 50) {
  bool allow_proceeding{false};
  if (this->task_launching_log_mode == 0) {
    allow_proceeding = false;
  } else if (this->task_launching_log_mode == 1 ||
      this->task_launching_log_mode == 2) {
    if (this->num_epochs == 1 || (this->num_epochs % (1 + term) == 0)) {
      // Allow log registration and its completion
      allow_proceeding = true;
    } else {
      allow_proceeding = false;
    }
  }
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
    // No more active tasks, stop the scheduler
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
