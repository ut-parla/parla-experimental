#include "include/phases.hpp"
#include "include/runtime.hpp"

/**************************/
// Mapper Implementation

void Mapper::enqueue(InnerTask *task) { this->mappable_tasks.push_back(task); }

void Mapper::enqueue(std::vector<InnerTask *> &tasks) {
  this->mappable_tasks.push_back(tasks);
}

size_t Mapper::get_count() {
  size_t count = this->mappable_tasks.atomic_size();
  return count;
}


void Mapper::run(SchedulerPhase *memory_reserver) {

  NVTX_RANGE("SpawnedPhase::run", NVTX_COLOR_LIGHT_GREEN)

  // TODO: Refactor this so its readable without as many nested conditionals

  // This is a non-critical region
  // Comment(wlr): Why is this a noncritical region?

  // Assumptions:
  // Scheduler maps a task to a device.
  // Scheduler does not reserve any resource at this phase.

  // TODO(hc): for now, I'm planning task mapping without policy.

  this->status.reset();

  bool has_task = true;

  /*
  has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask* task = this->spawned_tasks.front_and_pop();
    for (Device& dev : device_manager->GetAllDevices()) {
      if (dev.GetID() == dummy_dev_idx_ && dev.GetName().find("CUDA") != std::string::npos) {
        std::vector<std::vector<DeviceResources>> dev_req = {{DeviceResources{1000, 10}}};
        ResourceRequirement* req = new ResourceRequirement(std::move(dev_req));
        task->SetResourceRequirement(req);
        dummy_dev_idx_++;
      }
      if (dev.GetName().find("CPU") != std::string::npos) {

      }
    }
    this->mapped_tasks_buffer.push_back(task);
    has_task = this->get_count() > 0;
  } // while there are mappable tasks
  */

  // TODO:: Dummy implementation that just passes tasks through
  has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask *task = this->mappable_tasks.front_and_pop();
    this->mapped_tasks_buffer.push_back(task);
    has_task = this->get_count() > 0;
  }

  for (InnerTask *mapped_task : this->mapped_tasks_buffer) {

    mapped_task->notify_dependents(this->enqueue_buffer, Task::MAPPED);
    this->scheduler->enqueue_tasks(this->enqueue_buffer);
    this->enqueue_buffer.clear();

    bool enqueue_flag =
        (mapped_task->num_unreserved_dependencies.fetch_sub(1) == 1);

    if (enqueue_flag) {
      mapped_task->set_status(Task::RESERVABLE);
      memory_reserver->enqueue(mapped_task);
    }
  }

  this->mapped_tasks_buffer.clear();
}

/**************************/
// Reserved Phase implementation

void MemoryReserver::enqueue(InnerTask *task) {
  this->reservable_tasks.push_back(task);
}

void MemoryReserver::enqueue(std::vector<InnerTask *> &tasks) {
  this->reservable_tasks.push_back(tasks);
}

size_t MemoryReserver::get_count() {
  size_t count = this->reservable_tasks.atomic_size();
  return count;
}

void MemoryReserver::run(SchedulerPhase *runtime_reserver) {
  // Loop through all the tasks in the reservable_tasks queue, reserve memory on
  // device if possible;

  // TODO:: Dummy implementation that just passes tasks through
  bool has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask *task = this->reservable_tasks.front_and_pop();
    this->reserved_tasks_buffer.push_back(task);
    has_task = this->get_count() > 0;
  }

  for (InnerTask *reserved_task : this->reserved_tasks_buffer) {
    reserved_task->notify_dependents(this->enqueue_buffer, Task::RESERVED);
    this->scheduler->enqueue_tasks(this->enqueue_buffer);
    this->enqueue_buffer.clear();

    // TODO:(wlr) Create and possibly enqueue data movement tasks

    // Possibly enqueue this task
    bool enqueue_flag =
        (reserved_task->num_blocking_dependencies.fetch_sub(1) == 1);

    if (enqueue_flag) {
      reserved_task->set_status(Task::RUNNABLE);
      runtime_reserver->enqueue(reserved_task);
    }
  }

  this->reserved_tasks_buffer.clear();
}

/**************************/
// Ready Phase implementation

void RuntimeReserver::enqueue(InnerTask *task) {
  // std::cout << "Enqueuing task " << task->name << std::endl;
  this->runnable_tasks.push_back(task);
  // std::cout << "Ready tasks after enqueue: " <<
  // this->ready_tasks.atomic_size()
  //          << std::endl;
}

void RuntimeReserver::enqueue(std::vector<InnerTask *> &tasks) {
  // std::cout << "Enqueuing tasks " << tasks.size() << std::endl;
  // for (auto task : tasks) {
  //  this->enqueue(task);
  //}
  this->runnable_tasks.push_back(tasks);
  // std::cout << "Ready tasks after: " << this->ready_tasks.atomic_size()
  //          << std::endl;
}

size_t RuntimeReserver::get_count() {
  // std::cout << "Ready tasks: " << this->ready_tasks.atomic_size() <<
  // std::endl;
  size_t count = this->runnable_tasks.atomic_size();
  // std::cout << "Ready tasks: " << count << std::endl;
  return count;
}

void RuntimeReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("RuntimeReserver::run", NVTX_COLOR_LIGHT_GREEN)

  // TODO(wlr): Is this really the right way to handle this inheritance?
  Launcher *launcher = dynamic_cast<Launcher *>(next_phase);

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
      auto task = this->runnable_tasks.front();
      bool has_resources = scheduler->resources->check_greater(task->resources);

      if (has_resources) {

        bool has_thread = scheduler->workers.get_num_available_workers() > 0;

        if (has_thread) {

          InnerTask *task = this->runnable_tasks.front_and_pop();
          InnerWorker *worker = scheduler->workers.dequeue_worker();

          // Decrease Resources
          scheduler->resources->decrease(task->resources);

          launcher->enqueue(task, worker);

          this->status.increase(Ready::success);
        } else {
          this->status.increase(Ready::worker_miss);
          break; // No more workers available
        }
      } else {
        this->status.increase(Ready::resource_miss);
        break; // No more resources available
      }
    } else {
      this->status.increase(Ready::task_miss);
      break; // No more tasks available
    }
  }

  this->mtx.unlock();
}

/**************************/
// Launcher Phase implementation

void Launcher::enqueue(InnerTask *task, InnerWorker *worker) {
  NVTX_RANGE("Launcher::enqueue", NVTX_COLOR_LIGHT_GREEN)

  // Immediately launch task
  task->set_state(Task::RUNNING);
  this->num_running_tasks++;

  // Assign task to thread and notify via c++ condition variable.
  // No GIL needed until worker wakes.
  worker->assign_task(task);

  // std::cout << "Assigned " << task->name << " to " << worker->thread_idx
  //           << std::endl;
  LOG_INFO(WORKER, "Assigned {} to {}", task, worker);
}

void Launcher::run() {
  throw std::runtime_error("Launcher::run() not implemented.");
}
