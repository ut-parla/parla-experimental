#include "include/phases.hpp"
#include "include/runtime.hpp"

/**************************/
// Spawned Phase implementation

/*
void SpawnedPhase::run(MappedPhase *ready) {
  // NOT IMPLEMENTED
}
*/

void SpawnedPhase::enqueue(InnerTask *task) {
   std::cout << "[Spawned] Enqueuing task " << task->name << std::endl;
  this->spawned_tasks.push_back(task);
   std::cout << "[Spawned] Tasks after enqueue: " <<
   this->spawned_tasks.atomic_size()
            << std::endl;
}

void SpawnedPhase::enqueue(std::vector<InnerTask *> &tasks) {
  std::cout << "Enqueuing tasks " << tasks.size() << std::endl;
  // for (auto task : tasks) {
  //  this->enqueue(task);
  //}
  this->spawned_tasks.push_back(tasks);
  std::cout << "Ready tasks after: " << this->spawned_tasks.atomic_size()
            << std::endl;
}

size_t SpawnedPhase::get_count() {
  size_t count = this->spawned_tasks.atomic_size();
  return count;
}

void SpawnedPhase::run(ReadyPhase *ready_phase_handler) {
  NVTX_RANGE("SpawnedPhase::run", NVTX_COLOR_LIGHT_GREEN)

  // TODO: Refactor this so its readable without as many nested conditionals

  // This is a non-critical region

  // Assumptions:
  // Scheduler maps a task to a device.
  // Scheduler does not reserve any resource at this phase.

  // TODO(hc): for now, I'm planning task mapping without policy.

  this->status.reset();

  bool has_task = true;

  while (has_task) {

    has_task = this->get_count() > 0;

    /*
    if (has_task) {
      auto task = this->ready_tasks.front();
      bool has_resources = scheduler->resources->check_greater(task->resources);

      if (has_resources) {

        bool has_thread = scheduler->workers.get_num_available_workers() > 0;

        if (has_thread) {

          InnerTask *task = this->ready_tasks.front_and_pop();
          InnerWorker *worker = scheduler->workers.dequeue_worker();

          // Decrease Resources
          scheduler->resources->decrease(task->resources);

          launcher->enqueue(task, worker);

          this->status.update(Spawned::success);
        } else {
          this->status.update(Ready::worker_miss);
          break; // No more workers available
        }
      } else {
        this->status.update(Ready::resource_miss);
        break; // No more resources available
      }
    } else {
      this->status.update(Ready::task_miss);
      break; // No more tasks available
    }
    */
  }
}

/**************************/
// Mapped Phase implementation

void MappedPhase::run(ReservedPhase *ready) {
  // NOT IMPLEMENTED
}

/**************************/
// Reserved Phase implementation
void ReservedPhase::run(ReadyPhase *ready) {
  // NOT IMPLEMENTED
}

/**************************/
// Ready Phase implementation

void ReadyPhase::enqueue(InnerTask *task) {
  // std::cout << "Enqueuing task " << task->name << std::endl;
  this->ready_tasks.push_back(task);
  // std::cout << "Ready tasks after enqueue: " <<
  // this->ready_tasks.atomic_size()
  //          << std::endl;
}

void ReadyPhase::enqueue(std::vector<InnerTask *> &tasks) {
  // std::cout << "Enqueuing tasks " << tasks.size() << std::endl;
  // for (auto task : tasks) {
  //  this->enqueue(task);
  //}
  this->ready_tasks.push_back(tasks);
  // std::cout << "Ready tasks after: " << this->ready_tasks.atomic_size()
  //          << std::endl;
}

int ReadyPhase::get_count() {
  // std::cout << "Ready tasks: " << this->ready_tasks.atomic_size() <<
  // std::endl;
  int count = this->ready_tasks.atomic_size();
  // std::cout << "Ready tasks: " << count << std::endl;
  return count;
}

bool ReadyPhase::condition() {
  // NOT IMPLEMENTED
  return true;
}

void ReadyPhase::run(LauncherPhase *launcher) {
  NVTX_RANGE("ReadyPhase::run", NVTX_COLOR_LIGHT_GREEN)

  // TODO: Refactor this so its readable without as many nested conditionals

  // This is a critical region
  // Mutex needed only if it is called from multiple threads (not just scheduler
  // thread)

  // Assumptions:
  // Scheduler resources are ONLY decreased here
  // Available workers are ONLY decreased here

  // Assumptions to revisit:
  // Ready tasks must be launched in order
  // If the task at the head cannot be launched (eg. not enough resources, no
  // available workers) , then no other tasks can be launched
  // TODO: Revisit this design decision

  // TODO: Currently this drains the whole queue. Use Ready::condition() to set
  // a better policy?
  // TODO: This stops at a single failure.
  // TODO: Maybe failure of a phase means it should wait on events to try again.
  // Instead of just spinning?

  this->status.reset();

  this->mtx.lock();

  bool has_task = true;

  while (has_task) {

    has_task = this->get_count() > 0;

    if (has_task) {
      auto task = this->ready_tasks.front();
      bool has_resources = scheduler->resources->check_greater(task->resources);

      if (has_resources) {

        bool has_thread = scheduler->workers.get_num_available_workers() > 0;

        if (has_thread) {

          InnerTask *task = this->ready_tasks.front_and_pop();
          InnerWorker *worker = scheduler->workers.dequeue_worker();

          // Decrease Resources
          scheduler->resources->decrease(task->resources);

          launcher->enqueue(task, worker);

          this->status.update(Ready::success);
        } else {
          this->status.update(Ready::worker_miss);
          break; // No more workers available
        }
      } else {
        this->status.update(Ready::resource_miss);
        break; // No more resources available
      }
    } else {
      this->status.update(Ready::task_miss);
      break; // No more tasks available
    }
  }

  this->mtx.unlock();
}

/**************************/
// Launcher Phase implementation

void LauncherPhase::enqueue(InnerTask *task, InnerWorker *worker) {
  NVTX_RANGE("LauncherPhase::enqueue", NVTX_COLOR_LIGHT_GREEN)

  // Immediately launch task
  task->set_state(Task::running);
  this->num_running_tasks++;

  // std::cout << "Assigning " << task->name << " to " << worker->thread_idx
  //          << std::endl;

  /*
  //Acquire GIL to assign task to worker and notify worker through python
  callback void* py_task = task->py_task; void* py_worker = worker->py_worker;
  launch_task_callback(this->launch_callback, py_scheduler, py_task, py_worker);
  */

  // Assign task to thread and notify via c++ condition variable. No GIL needed
  // until worker wakes.
  worker->assign_task(task);

  // std::cout << "Assigned " << task->name << " to " << worker->thread_idx
  //           << std::endl;
  LOG_INFO(WORKER, "Assigned {} to {}", task, worker);
}

void LauncherPhase::run() {
  // NOT IMPLEMENTED
}
