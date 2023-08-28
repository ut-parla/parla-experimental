#include "include/containers.hpp"
#include "include/resources.hpp"
#include "include/runtime.hpp"
#include <string.h>

#define DEPENDENCY_BUFFER_SIZE 4

// Task Implementation

// TODO(hc) member initialization list is preferable as it can reduce
//          instructions (e.g.,
//          https://stackoverflow.com/questions/9903248/initializing-fields-in-constructor-initializer-list-vs-constructor-body)
InnerTask::InnerTask()
    : req_addition_mode_(SingleDevAdd), tmp_arch_req_(nullptr),
      tmp_multdev_reqs_(nullptr) {
  this->dependency_buffer.reserve(DEPENDENCY_BUFFER_SIZE);
  this->id = 0;
  this->py_task = nullptr;
}

InnerTask::InnerTask(long long int id, void *py_task)
    : req_addition_mode_(SingleDevAdd) {
  this->dependency_buffer.reserve(DEPENDENCY_BUFFER_SIZE);
  this->id = id;
  this->py_task = py_task;
}

InnerTask::InnerTask(std::string name, long long int id, void *py_task)
    : req_addition_mode_(SingleDevAdd) {
  this->dependency_buffer.reserve(DEPENDENCY_BUFFER_SIZE);
  this->name = name;
  this->id = id;
  this->py_task = py_task;
}

void InnerTask::set_scheduler(InnerScheduler *scheduler) {
  this->scheduler = scheduler;
  size_t num_devices = this->scheduler->get_device_manager()->get_num_devices();
  this->parray_list.resize(num_devices);
}

void InnerTask::set_name(std::string name) {
  // std::cout << "Setting name to " << name << std::endl;
  this->name = name;
}

void InnerTask::set_id(long long int id) { this->id = id; }

void InnerTask::set_py_task(void *py_task) { this->py_task = py_task; }

void InnerTask::set_priority(int priority) { this->priority = priority; }

void InnerTask::queue_dependency(InnerTask *task) {
  this->dependency_buffer.push_back(task);
}

TaskStatusFlags InnerTask::process_dependencies() {
  NVTX_RANGE("InnerTask::process_dependencies", NVTX_COLOR_MAGENTA)
  TaskStatusFlags status = this->add_dependencies(this->dependency_buffer);
  this->dependency_buffer.clear();
  this->dependency_buffer.reserve(DEPENDENCY_BUFFER_SIZE);
  return status;
}

void InnerTask::clear_dependencies() {
  this->dependency_buffer.clear();
  this->dependencies.clear();
  // this->dependency_buffer.reserve(DEPENDENCY_BUFFER_SIZE);
}

TaskState InnerTask::add_dependency(InnerTask *task) {

  // Store all added dependencies for bookkeeping
  // I cannot think of a scenario when multiple writers would be adding
  // dependencies NOTE: Please make this thread safe if we have one
  this->dependencies.push_back_unsafe(task);

  // If the task is already complete, we don't need to add it to the dependency

  bool dependency_complete = false;

  TaskState dependent_state = task->add_dependent_task(this);

  if (dependent_state >= TaskState::RUNAHEAD) {
    this->num_blocking_dependencies.fetch_sub(1);
    this->num_unspawned_dependencies.fetch_sub(1);
    this->num_unmapped_dependencies.fetch_sub(1);
    this->num_unreserved_dependencies.fetch_sub(1);
  } else if (dependent_state >= TaskState::RESERVED) {
    this->num_unspawned_dependencies.fetch_sub(1);
    this->num_unmapped_dependencies.fetch_sub(1);
    this->num_unreserved_dependencies.fetch_sub(1);
  } else if (dependent_state >= TaskState::MAPPED) {
    this->num_unspawned_dependencies.fetch_sub(1);
    this->num_unmapped_dependencies.fetch_sub(1);
  } else if (dependent_state >= TaskState::SPAWNED) {
    this->num_unspawned_dependencies.fetch_sub(1);
  }

  return dependent_state;
}

TaskStatus InnerTask::determine_status(bool new_spawnable, bool new_mappable,
                                       bool new_reservable, bool new_runnable) {
  if (new_runnable and this->processed_data) {
    return TaskStatus::RUNNABLE;
  } else if (new_runnable and !this->processed_data) {
    return TaskStatus::COMPUTE_RUNNABLE;
  } else if (new_reservable) {
    return TaskStatus::RESERVABLE;
  } else if (new_mappable) {
    return TaskStatus::MAPPABLE;
  } else if (new_spawnable) {
    return TaskStatus::SPAWNABLE;
  } else {
    return TaskStatus::INITIAL;
  }
}

TaskStatusFlags InnerTask::add_dependencies(std::vector<InnerTask *> &tasks,
                                            bool data_tasks) {
  LOG_INFO(TASK, "Adding dependencies to {}. D={}", this, tasks);

  // TODO: Change all of this to lock free.
  //       Handle phase events

  if (data_tasks == false) {
    this->num_unspawned_dependencies.fetch_add(tasks.size());
    this->num_unmapped_dependencies.fetch_add(tasks.size());
    this->num_unreserved_dependencies.fetch_add(tasks.size());
    this->num_blocking_compute_dependencies.fetch_add(tasks.size());
  }

  // CHECKME: Is this still correct on the subsequent call (should be
  // tasks.size()+1 to prevent finishing while adding)
  this->num_blocking_dependencies.fetch_add(tasks.size());

  for (size_t i = 0; i < tasks.size(); i++) {
    this->add_dependency(tasks[i]);
  }

  // Decrement overcount to free this region
  bool spawnable = this->num_unspawned_dependencies.fetch_sub(1) == 1;
  bool mappable = this->num_unmapped_dependencies.fetch_sub(1) == 1;

  // Other counters are 'freed' in each phase before entering the next phase

  TaskStatusFlags status = TaskStatusFlags();
  status.spawnable = spawnable;
  status.mappable = mappable;

  // TaskStatus status =
  //     this->determine_status(spawnable, mappable, reservable, ready);

  LOG_INFO(TASK, "Added dependencies to {}. Status = {}", this, status);

  return status;

  // If true, this task is ready to launch. Launching must be handled
  // Otherwise launching will be handled by another task's notify_dependents
}

/*
 *    let n = number of dependencies
 *    want to decrement n such that n = n-1 when both threads have run
 *
 *    thread 1                       thread 2
 *    s1 = check complete
 *    s1  ? n-- : n++
 *                                   notify_dependents: noop  -> s3
 *    add to dependents
 *                                   notify_dependents: n--   -> s4
 *    s2 = check complete
 *    s1 && s1 ? noop : n--
 *
 *    This handles all cases except when s1=0 & s3=1
 *    Adding lock to ensure s1 == s3, reverts to what we had before:
 *
 *    thread 1                       thread 2
 *                                   notify_dependents_mutex(): noop
 *    mutex_lock()
 *    s1 = check complete
 *    add to dependents
 *    mutex_unlock()
 *    s1  ? n-- : noop
 *                                   notify_dependents_mutex(): n--
 *    s2 = check complete
 *
 *    I am sure there is a better implementation of this.
 */

TaskState InnerTask::add_dependent_task(InnerTask *task) {

  // Store all dependents for bookkeeping
  // Dependents can be written to by multiple threads calling this function
  // Dependents is read when the task is in cleanup, which can overlap with this
  // function This write needs to be thread-safe

  // NOTE: This is not a lock free implementation. I don't know how to make it
  // lock free
  //       This lock and its match in notify_dependents ensures correctness,
  //       explained above. A tasks "completeness" cannot change while we are
  //       adding a dependent to the list This means notify dependents has not
  //       run yet.
  this->dependents.lock();

  TaskState state = this->get_state();     // s1
  this->dependents.push_back_unsafe(task); // s3

  this->dependents.unlock();

  return state;
}

TaskState InnerTask::add_dependent_space(TaskBarrier *barrier) {
  this->spaces.lock();
  TaskState state = this->get_state();
  this->spaces.push_back_unsafe(barrier);
  this->spaces.unlock();

  return state;
}

void InnerTask::add_parray(parray::InnerPArray *parray, int am, int dev_id) {
  AccessMode access_mode = static_cast<AccessMode>(am);
  if (access_mode != AccessMode::IN) {
    parray->get_parent_parray()->add_task(this);
  }
  this->parray_list[dev_id].emplace_back(std::make_pair(parray, access_mode));
}

void InnerTask::notify_dependents_completed() {
  LOG_INFO(TASK, "Notifying dependents of {}.", this);
  NVTX_RANGE("InnerTask::notify_dependents", NVTX_COLOR_MAGENTA)

  this->dependents.lock();
  this->spaces.lock();

  for (size_t i = 0; i < this->spaces.size_unsafe(); i++) {
    auto space = this->spaces.get_unsafe(i);
    space->notify();
  }

  this->set_state(TaskState::COMPLETED);

  this->spaces.unlock();
  this->dependents.unlock();

  LOG_INFO(TASK, "Notified dependents of {}.", this);
}

void InnerTask::notify_dependents(TaskStatusList &buffer, TaskState new_state) {
  LOG_INFO(TASK, "Notifying dependents of {}: {}", this, buffer);
  NVTX_RANGE("InnerTask::notify_dependents", NVTX_COLOR_MAGENTA)

  // NOTE: I changed this to queue up ready tasks instead of enqueing them one
  // at a time
  //       This is possibly worse, but splits out scheduler dependency.
  //       May need to change back to call scheduler.enqueue(task) here instead

  this->dependents.lock();
  // std::cout << "Notifying dependents of " << this->name << ": " <<
  // this->dependents.size_unsafe() << std::endl;

  for (size_t i = 0; i < this->dependents.size_unsafe(); i++) {

    auto task = this->dependents.get_unsafe(i);
    TaskStatusFlags status = task->notify(new_state, this->is_data.load());

    // std::cout << "Dependent Task is notified: " << task->name << std::endl;
    if (status.any()) {
      // std::cout << "Dependent Task Ready: " << task->name << std::endl;
      buffer.push_back(std::make_pair(task, status));
    }
  }

  this->set_state(new_state);

  this->dependents.unlock();

  // std::cout << "Notified dependents of " << this->name << ". Ready tasks: "
  // << buffer.size() << std::endl;

  LOG_INFO(TASK, "Notified dependents of {}. Ready tasks: {}", this, buffer);
}

bool InnerTask::notify_dependents_wrapper() {
  TaskStatusList buffer = TaskStatusList();
  this->notify_dependents(buffer, TaskState::MAPPED);
  return buffer.size() > 0;
}

TaskStatusFlags InnerTask::notify(TaskState dependency_state, bool is_data) {

  bool spawnable = false;
  bool mappable = false;
  bool reservable = false;
  bool compute_runnable = false;
  bool runnable = false;

  if (is_data) {
    if (dependency_state == TaskState::RUNAHEAD) {
      // A data task never notifies for the other stages
      runnable = (this->num_blocking_dependencies.fetch_sub(1) == 1);
    }
  } else {
    if (dependency_state == TaskState::RUNAHEAD) {
      compute_runnable =
          (this->num_blocking_compute_dependencies.fetch_sub(1) == 1);
      runnable = (this->num_blocking_dependencies.fetch_sub(1) == 1);
    } else if (dependency_state >= TaskState::RESERVED) {
      reservable = (this->num_unreserved_dependencies.fetch_sub(1) == 1);
    } else if (dependency_state >= TaskState::MAPPED) {
      mappable = (this->num_unmapped_dependencies.fetch_sub(1) == 1);
    } else if (dependency_state >= TaskState::SPAWNED) {
      spawnable = (this->num_unspawned_dependencies.fetch_sub(1) == 1);
    }
  }

  TaskStatusFlags status;
  status.spawnable = spawnable;
  status.mappable = mappable;
  status.reservable = reservable;
  status.compute_runnable = compute_runnable;
  status.runnable = runnable;

  return status;
}

bool InnerTask::blocked() { return this->num_blocking_dependencies.load() > 0; }

std::string InnerTask::get_name() { return this->name; }

int InnerTask::get_num_dependencies() {
  return this->dependencies.atomic_size();
}

int InnerTask::get_num_dependents() { return this->dependents.atomic_size(); }

std::vector<void *> InnerTask::get_dependencies() {
  std::vector<void *> dependency_list;
  this->dependencies.lock();
  for (size_t i = 0; i < this->dependencies.size_unsafe(); i++) {
    dependency_list.push_back(this->dependencies.get_unsafe(i));
  }
  this->dependencies.unlock();
  return dependency_list;
}

std::vector<void *> InnerTask::get_dependents() {
  std::vector<void *> dependent_list;
  this->dependents.lock();
  for (size_t i = 0; i < this->dependents.size_unsafe(); i++) {
    dependent_list.push_back(this->dependents.get_unsafe(i));
  }
  this->dependents.unlock();
  return dependent_list;
}

void *InnerTask::get_py_task() { return this->py_task; }

int InnerTask::set_state(int state) {
  TaskState new_state = static_cast<TaskState>(state);
  TaskState old_state = this->set_state(new_state);
  int old_state_id = static_cast<int>(new_state);
  return old_state_id;
}

std::vector<Device *> &InnerTask::get_assigned_devices() {
  return this->assigned_devices;
}

void InnerTask::copy_assigned_devices(const std::vector<Device *> &others) {
  this->assigned_devices = others;
}

void InnerTask::add_assigned_device(Device *device) {
  this->assigned_devices.push_back(device);
}

TaskState InnerTask::set_state(TaskState state) {
  TaskState new_state = state;
  TaskState old_state;

  do {
    old_state = this->state.load();
  } while (!this->state.compare_exchange_weak(old_state, new_state));

  return old_state;
}

TaskStatus InnerTask::set_status(TaskStatus status) {
  TaskStatus new_status = status;
  TaskStatus old_status;

  do {
    old_status = this->status.load();
  } while (!this->status.compare_exchange_weak(old_status, new_status));

  return old_status;
}

/*TODO(wlr): Deprecate this before merge. Need to update pxd and tests*/
void InnerTask::set_complete() { this->set_state(TaskState::COMPLETED); }

bool InnerTask::get_complete() {
  return this->get_state() == TaskState::COMPLETED;
}

// TODO(hc): The current Parla exploits two types of resources,
//           memory and vcus. Later, this can be extended with
//           a map.
void InnerTask::add_device_req(Device *dev_ptr, MemorySz_t mem_sz,
                               VCU_t num_vcus) {
  ResourcePool_t res_req;
  res_req.set(Resource::Memory, mem_sz);
  res_req.set(Resource::VCU, num_vcus);

  std::shared_ptr<DeviceRequirement> dev_req =
      std::make_shared<DeviceRequirement>(dev_ptr, res_req);
  if (req_addition_mode_ == SingleDevAdd) {
    placement_req_options_.append_placement_req_opt(std::move(dev_req));
  } else if (req_addition_mode_ % 2 == 0) { /* Architecture requirement */
    tmp_arch_req_->append_placement_req_opt(std::move(dev_req));
  } else if (req_addition_mode_ == MultiDevAdd) {
    tmp_multdev_reqs_->append_placement_req(std::move(dev_req));
  }
}

void InnerTask::begin_arch_req_addition() {
  // Setting architecture resource requirement
  // could be called within multi-device requirement
  // setup.
  ++req_addition_mode_;
  assert(tmp_arch_req_ == nullptr);
  tmp_arch_req_ = std::make_shared<ArchitectureRequirement>();
}

void InnerTask::end_arch_req_addition() {
  assert(req_addition_mode_ % 2 == 0);
  if (req_addition_mode_ == 4) {
    tmp_multdev_reqs_->append_placement_req(std::move(tmp_arch_req_));
  } else {
    placement_req_options_.append_placement_req_opt(std::move(tmp_arch_req_));
  }
  --req_addition_mode_;
}

void InnerTask::begin_multidev_req_addition() {
  assert(req_addition_mode_ == SingleDevAdd);
  assert(tmp_multdev_reqs_ == nullptr);
  tmp_multdev_reqs_ = std::make_shared<MultiDeviceRequirements>();
  req_addition_mode_ = MultiDevAdd;
}

void InnerTask::end_multidev_req_addition() {
  assert(tmp_multdev_reqs_ != nullptr);
  placement_req_options_.append_placement_req_opt(std::move(tmp_multdev_reqs_));
  req_addition_mode_ = SingleDevAdd;
}

void *InnerDataTask::get_py_parray() { return this->parray_->get_py_parray(); }

int InnerDataTask::get_access_mode() {
  return static_cast<int>(this->access_mode_);
}

TaskState TaskBarrier::_add_task(InnerTask *task) {
  TaskState dependent_state = task->add_dependent_space(this);

  if (dependent_state == TaskState::COMPLETED) {
    this->num_incomplete_tasks.fetch_sub(1, std::memory_order_relaxed);
  }

  return dependent_state;
}

void TaskBarrier::add_task(InnerTask *task) {
  // don't use this
  this->num_incomplete_tasks.fetch_add(1, std::memory_order_relaxed);
  this->_add_task(task);
  this->notify();
}

void TaskBarrier::add_tasks(std::vector<InnerTask *> &tasks) {

  this->num_incomplete_tasks.fetch_add(tasks.size() + 1,
                                       std::memory_order_relaxed);
  for (auto task : tasks) {
    this->_add_task(task);
  }

  // std::cout << "TaskBarrier::add_tasks: " <<
  // this->num_incomplete_tasks.load()
  //           << std::endl;

  this->notify();
}