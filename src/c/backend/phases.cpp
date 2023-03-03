#include "include/phases.hpp"
#include "include/profiling.hpp"
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

  NVTX_RANGE("Mapper::run", NVTX_COLOR_LIGHT_GREEN)

  // TODO: Refactor this so its readable without as many nested conditionals

  // This is a non-critical region
  // Comment(wlr): Why is this a noncritical region?

  // Assumptions:
  // Scheduler maps a task to a device.
  // Scheduler does not reserve any resource at this phase.

  // TODO(hc): for now, I'm planning task mapping without policy.

  this->status.reset();

  bool has_task = true;

  has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask *task = this->mappable_tasks.front_and_pop();
    ResourceRequirementCollections &res_reqs = task->GetResourceRequirements();
    std::vector<DeviceRequirementBase *> dev_res_reqs =
        res_reqs.GetDeviceRequirementOptions();
    for (DeviceRequirementBase *r : dev_res_reqs) {
      if (r->is_multidev_req()) {
        // TODO(hc): It can be refactored later and its length
        //           can be reduced.
        //           Refactor it when we implement a policy.
        std::cout << "[Multi-device requirement]\n";
        MultiDeviceRequirements *mdev_res_reqs =
            dynamic_cast<MultiDeviceRequirements *>(r);
        const std::vector<SingleDeviceRequirementBase *> mdev_res_reqs_vec =
            mdev_res_reqs->GetDeviceRequirements();
        for (DeviceRequirementBase *m_r : mdev_res_reqs_vec) {
          if (m_r->is_dev_req()) {
            DeviceRequirement *dev_res_req =
                dynamic_cast<DeviceRequirement *>(m_r);
            std::cout << "\t[Device Requirement in Multi-device Requirement]\n";
            std::cout << "\t" << dev_res_req->device().GetName() << " -> "
                      << dev_res_req->res_req().mem_sz << "B, VCU "
                      << dev_res_req->res_req().num_vcus << "\n";
          } else if (m_r->is_arch_req()) {
            std::cout
                << "\t[Architecture Requirement in Multi-device Requirement]\n";
            ArchitectureRequirement *arch_res_req =
                dynamic_cast<ArchitectureRequirement *>(m_r);
            uint32_t i = 0;
            for (DeviceRequirement *dev_res_req :
                 arch_res_req->GetDeviceRequirementOptions()) {
              std::cout << "\t\t[" << i << "]"
                        << dev_res_req->device().GetName() << " -> "
                        << dev_res_req->res_req().mem_sz << "B, VCU "
                        << dev_res_req->res_req().num_vcus << "\n";
              ++i;
            }
          }
        }
      } else if (r->is_dev_req()) {
        DeviceRequirement *dev_res_req = dynamic_cast<DeviceRequirement *>(r);
        std::cout << "[Device Requirement]\n";
        std::cout << dev_res_req->device().GetName() << " -> "
                  << dev_res_req->res_req().mem_sz << "B, VCU "
                  << dev_res_req->res_req().num_vcus << "\n";
      } else if (r->is_arch_req()) {
        std::cout << "[Architecture Requirement]\n";
        ArchitectureRequirement *arch_res_req =
            dynamic_cast<ArchitectureRequirement *>(r);
        uint32_t i = 0;
        for (DeviceRequirement *dev_res_req :
             arch_res_req->GetDeviceRequirementOptions()) {
          std::cout << "\t[" << i << "]" << dev_res_req->device().GetName()
                    << " -> " << dev_res_req->res_req().mem_sz << "B, VCU "
                    << dev_res_req->res_req().num_vcus << "\n";
          ++i;
        }
      }
    }

    this->mapped_tasks_buffer.push_back(task);
    has_task = this->get_count() > 0;
  } // while there are mappable tasks

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
  NVTX_RANGE("MemoryReserver::run", NVTX_COLOR_LIGHT_GREEN)
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
