#include "include/containers.hpp"
#include "include/runtime.hpp"
#include <string.h>

#define DEPENDENCY_BUFFER_SIZE 10

// Task Implementation

InnerTask::InnerTask() {
  this->dependency_buffer.reserve(DEPENDENCY_BUFFER_SIZE);
  this->id = 0;
  this->py_task = nullptr;
}

InnerTask::InnerTask(long long int id, void *py_task) {
  this->dependency_buffer.reserve(DEPENDENCY_BUFFER_SIZE);
  this->id = id;
  this->py_task = py_task;
}

InnerTask::InnerTask(std::string name, long long int id, void *py_task) {
  this->dependency_buffer.reserve(DEPENDENCY_BUFFER_SIZE);
  this->name = name;
  this->id = id;
  this->py_task = py_task;
}

void InnerTask::set_scheduler(InnerScheduler *scheduler) {
  this->scheduler = scheduler;
}

void InnerTask::set_name(std::string name) {
  // std::cout << "Setting name to " << name << std::endl;
  this->name = name;
}

void InnerTask::set_id(long long int id) { this->id = id; }

void InnerTask::set_py_task(void *py_task) { this->py_task = py_task; }

void InnerTask::set_priority(int priority) { this->priority = priority; }

void InnerTask::set_resources(std::string resource_name, float resource_value) {
  this->resources.set(resource_name, resource_value);
}

void InnerTask::queue_dependency(InnerTask *task) {
  this->dependency_buffer.push_back(task);
}

Task::StatusFlags InnerTask::process_dependencies() {
  NVTX_RANGE("InnerTask::process_dependencies", NVTX_COLOR_MAGENTA)
  Task::StatusFlags status = this->add_dependencies(this->dependency_buffer);
  this->dependency_buffer.clear();
  return status;
}

void InnerTask::clear_dependencies() {
  this->dependency_buffer.clear();
  this->dependencies.clear();
}

Task::State InnerTask::add_dependency(InnerTask *task) {

  // Store all added dependencies for bookkeeping
  // I cannot think of a scenario when multiple writers would be adding
  // dependencies NOTE: Please make this thread safe if we have one
  this->dependencies.push_back_unsafe(task);

  // If the task is already complete, we don't need to add it to the dependency

  bool dependency_complete = false;

  Task::State dependent_state = task->add_dependent(this);

  if (dependent_state >= Task::RUNAHEAD) {
    this->num_blocking_dependencies.fetch_sub(1);
    this->num_unspawned_dependencies.fetch_sub(1);
    this->num_unmapped_dependencies.fetch_sub(1);
    this->num_unreserved_dependencies.fetch_sub(1);
  } else if (dependent_state >= Task::RESERVED) {
    this->num_unspawned_dependencies.fetch_sub(1);
    this->num_unmapped_dependencies.fetch_sub(1);
    this->num_unreserved_dependencies.fetch_sub(1);
  } else if (dependent_state >= Task::MAPPED) {
    this->num_unspawned_dependencies.fetch_sub(1);
    this->num_unmapped_dependencies.fetch_sub(1);
  } else if (dependent_state >= Task::SPAWNED) {
    this->num_unspawned_dependencies.fetch_sub(1);
  }

  return dependent_state;
}

Task::Status InnerTask::determine_status(bool new_spawnable, bool new_mappable,
                                         bool new_reservable,
                                         bool new_runnable) {
  if (new_runnable and this->processed_data) {
    return Task::RUNNABLE;
  } else if (new_runnable and !this->processed_data) {
    return Task::COMPUTE_RUNNABLE;
  } else if (new_reservable) {
    return Task::RESERVABLE;
  } else if (new_mappable) {
    return Task::MAPPABLE;
  } else if (new_spawnable) {
    return Task::SPAWNABLE;
  } else {
    return Task::INITIAL;
  }
}

Task::StatusFlags InnerTask::add_dependencies(std::vector<InnerTask *> &tasks) {

  bool data_tasks = false;

  LOG_INFO(TASK, "Adding dependencies to {}. D={}", this, tasks);

  // TODO: Change all of this to lock free.
  //       Handle phase events

  if (data_tasks == false) {
    this->num_unspawned_dependencies.fetch_add(tasks.size());
    this->num_unmapped_dependencies.fetch_add(tasks.size());
    this->num_unreserved_dependencies.fetch_add(tasks.size());
    this->num_blocking_compute_dependencies.fetch_add(tasks.size());
  }
  this->num_blocking_dependencies.fetch_add(tasks.size());

  for (size_t i = 0; i < tasks.size(); i++) {
    this->add_dependency(tasks[i]);
  }

  // Decrement overcount to free this region
  bool spawnable = this->num_unspawned_dependencies.fetch_sub(1) == 1;
  bool mappable = this->num_unspawned_dependencies.fetch_sub(1) == 1;

  // Other counters are 'freed' in each phase before entering the next phase

  Task::StatusFlags status = Task::StatusFlags();
  status.spawnable = spawnable;
  status.mappable = mappable;

  // Task::Status status =
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

Task::State InnerTask::add_dependent(InnerTask *task) {

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

  Task::State state = this->get_state();   // s1
  this->dependents.push_back_unsafe(task); // s3

  this->dependents.unlock();

  return state;
}

void InnerTask::notify_dependents(TaskStateList &buffer,
                                  Task::State new_state) {
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
    Task::StatusFlags status = task->notify(new_state);

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
  TaskStateList buffer = TaskStateList();
  this->notify_dependents(buffer, Task::COMPLETED);
  return buffer.size() > 0;
}

Task::StatusFlags InnerTask::notify(Task::State dependency_state,
                                    bool is_data) {

  bool spawnable = false;
  bool mappable = false;
  bool reservable = false;
  bool compute_runnable = false;
  bool runnable = false;

  if (is_data) {
    if (dependency_state >= Task::RUNAHEAD) {
      // A data task never notifies for the other stages
      runnable = (this->num_blocking_dependencies.fetch_sub(1) == 1);
    }
  } else {
    if (dependency_state >= Task::RUNAHEAD) {
      compute_runnable ==
          (this->num_blocking_compute_dependencies.fetch_sub(1) == 1);
      runnable = (this->num_blocking_dependencies.fetch_sub(1) == 1);
    } else if (dependency_state >= Task::RESERVED) {
      reservable = (this->num_unreserved_dependencies.fetch_sub(1) == 1);
    } else if (dependency_state >= Task::MAPPED) {
      mappable = (this->num_unmapped_dependencies.fetch_sub(1) == 1);
    } else if (dependency_state >= Task::SPAWNED) {
      spawnable = (this->num_unspawned_dependencies.fetch_sub(1) == 1);
    }
  }

  Task::StatusFlags status;
  status.spawnable = spawnable;
  status.mappable = mappable;
  status.reservable = reservable;
  status.compute_runnable = compute_runnable;
  status.runnable = runnable;

  return status;
}

bool InnerTask::blocked() { return this->num_blocking_dependencies.load() > 0; }

int InnerTask::get_num_dependencies() {
  return this->dependencies.atomic_size();
}

int InnerTask::get_num_dependents() { return this->dependents.atomic_size(); }

int InnerTask::get_num_blocking_dependencies() const {
  return this->num_blocking_dependencies.load();
}

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
  Task::State new_state = static_cast<Task::State>(state);
  Task::State old_state = this->set_state(new_state);
  int old_state_id = static_cast<int>(new_state);
  return old_state_id;
}

Task::State InnerTask::set_state(Task::State state) {
  Task::State new_state = state;
  Task::State old_state;

  do {
    old_state = this->state.load();
  } while (!this->state.compare_exchange_weak(old_state, new_state));

  return old_state;
}

Task::Status InnerTask::set_status(Task::Status status) {
  Task::Status new_status = status;
  Task::Status old_status;

  do {
    old_status = this->status.load();
  } while (!this->status.compare_exchange_weak(old_status, new_status));

  return old_status;
}

/*TODO(wlr): Deprecate this before merge.Need to update pxd and tests*/
void InnerTask::set_complete() { this->set_state(Task::COMPLETED); }

bool InnerTask::get_complete() { return this->get_state(); }
