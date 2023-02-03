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

void InnerTask::set_name(std::string name) { this->name = name; }

void InnerTask::set_id(long long int id) { this->id = id; }

void InnerTask::set_py_task(void *py_task) { this->py_task = py_task; }

void InnerTask::set_priority(int priority) { this->priority = priority; }

void InnerTask::set_resources(std::string resource_name, float resource_value) {
  this->resources.set(resource_name, resource_value);
}

void InnerTask::queue_dependency(InnerTask *task) {
  this->dependency_buffer.push_back(task);
}

bool InnerTask::process_dependencies() {
  my_scoped_range r("task::process_dependencies", nvtx3::rgb{127, 127, 0});
  // std::cout << "Processing Dependencies: " << this-> name << " " <<
  // this->dependency_buffer.size() << std::endl;
  bool status = this->add_dependencies(this->dependency_buffer);
  this->dependency_buffer.clear();
  return status;
}

void InnerTask::clear_dependencies() {
  this->dependency_buffer.clear();
  this->dependencies.clear();
}

bool InnerTask::add_dependency(InnerTask *task) {

  // Store all added dependencies for bookkeeping
  // I cannot think of a scenario when multiple writers would be adding
  // dependencies NOTE: Please make this thread safe if we have one
  this->dependencies.push_back_unsafe(task);

  // If the task is already complete, we don't need to add it to the dependency
  // count

  if (task->add_dependent(this)) {
    this->num_blocking_dependencies--;

    // Return that the dependency was already complete
    return true;
  }

  // Return that the dependency was not complete
  return false;
}

bool InnerTask::add_dependencies(std::vector<InnerTask *> &tasks) {

  // TODO: This will need to include all other dependency trackers
  // (num_mapped_dependencies, etc)

  // TODO: num_spawned_dependencies should be handled before this stage as this
  // tasks predecessors need to exist
  //       unless we change to creating a task object on first taskspace
  //       reference instead of at spawn time. Something to consider.

  this->num_blocking_dependencies.store(tasks.size() + 1);

  for (size_t i = 0; i < tasks.size(); i++) {
    this->add_dependency(tasks[i]);
  }

  int before_value = this->num_blocking_dependencies.fetch_sub(1);

  if (before_value == 1) {
    // This task is ready to run
    // Launching must be handled
    return true;
  }

  // The task was not ready to run during construction
  // Launching will be handled by another task's notify_dependents
  return false;
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

bool InnerTask::add_dependent(InnerTask *task) {
  my_scoped_range r("task::add_dependents", nvtx3::rgb{127, 127, 0});
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

  bool state = this->complete;             // s1
  this->dependents.push_back_unsafe(task); // s3

  this->dependents.unlock();

  if (state) {
    // Dependency was already complete, so we need to decrement the dependency
    // count
    return true;
  }
  // Dependency was not complete before adding to dependents list.
  return false;
}

std::vector<InnerTask *> &
InnerTask::notify_dependents(std::vector<InnerTask *> &buffer) {
  my_scoped_range r("task::notify_dependents", nvtx3::rgb{127, 127, 0});
  // NOTE: I changed this to queue up ready tasks instead of enqueing them one
  // at a time
  //       This is possibly worse, but splits out scheduler dependency.
  //       May need to change back to call scheduler.enqueue(task) here instead

  this->dependents.lock();
  // std::cout << "Notifying dependents of " << this->name << ": " <<
  // this->dependents.size_unsafe() << std::endl;

  for (size_t i = 0; i < this->dependents.size_unsafe(); i++) {
    auto task = this->dependents.get_unsafe(i);
    bool ready = task->notify();

    if (ready) {
      // std::cout << "Dependent Task Ready" << std::endl;
      buffer.push_back(task);
    }
  }

  this->set_complete(true);
  this->dependents.unlock();

  return buffer;
}

bool InnerTask::notify_dependents_wrapper() {
  std::vector<InnerTask *> buffer = std::vector<InnerTask *>();
  this->notify_dependents(buffer);
  return buffer.size() > 0;
}

bool InnerTask::notify() {
  int remaining = this->num_blocking_dependencies.fetch_sub(1) - 1;

  if (remaining == 0) {
    // This task is now ready to run
    return true;
  }
  // This task has more dependencies to wait on
  return false;
}

bool InnerTask::blocked() { return this->num_blocking_dependencies.load() > 0; }

int InnerTask::get_num_dependencies() {
  return this->dependencies.atomic_size();
}

int InnerTask::get_num_dependents() { return this->dependents.atomic_size(); }

int InnerTask::get_num_blocking_dependencies() {
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

void InnerTask::set_state(int state) {
  Task::State new_state = (Task::State)state;
  this->state.store(new_state);
}

void InnerTask::set_state(Task::State state) { this->state.store(state); }

void InnerTask::set_complete(bool complete) { this->complete.store(complete); }

bool InnerTask::get_complete() { return this->complete.load(); }
