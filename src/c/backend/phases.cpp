#include "include/phases.hpp"
#include "include/device.hpp"
#include "include/policy.hpp"
#include "include/profiling.hpp"
#include "include/resources.hpp"
#include "include/runtime.hpp"
#include <algorithm>

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

  // TODO(wlr): This is just used for random testing policy
  //            Remove this when we implement a policy.
  std::vector<Device *> devices(
      this->device_manager->get_devices(DeviceType::ANY));

  has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask *task = this->mappable_tasks.front_and_pop();

    /*
    ResourceRequirementCollections &res_reqs = task->GetResourceRequirements();
    std::vector<std::shared_ptr<DeviceRequirementBase>> dev_res_reqs =
        res_reqs.GetDeviceRequirementOptions();
    for (std::shared_ptr<DeviceRequirementBase> r : dev_res_reqs) {
      if (r->is_multidev_req()) {
        // TODO(hc): It can be refactored later and its length
        //           can be reduced.
        //           Refactor it when we implement a policy.
        std::cout << "[Multi-device requirement]\n";
        MultiDeviceRequirements *mdev_res_reqs =
            dynamic_cast<MultiDeviceRequirements *>(r.get());
        const std::vector<std::shared_ptr<SingleDeviceRequirementBase>>
            mdev_res_reqs_vec = mdev_res_reqs->GetDeviceRequirements();
        for (std::shared_ptr<DeviceRequirementBase> m_r : mdev_res_reqs_vec) {
          if (m_r->is_dev_req()) {
            DeviceRequirement *dev_res_req =
                dynamic_cast<DeviceRequirement *>(m_r.get());
            policy_->MapTask(task, *(dev_res_req->device()));
            std::cout << "\t[Device Requirement in Multi-device "
                         "Requirement]\n";
            std::cout << "\t" << dev_res_req->device()->get_name() << " -> "
                      << dev_res_req->res_req().get(MEMORY) << "B, VCU "
                      << dev_res_req->res_req().get(VCU) << "\n";
          } else if (m_r->is_arch_req()) {
            std::cout << "\t[Architecture Requirement in "
                         "Multi-device Requirement]\n";
            ArchitectureRequirement *arch_res_req =
                dynamic_cast<ArchitectureRequirement *>(m_r.get());
            uint32_t i = 0;
            for (std::shared_ptr<DeviceRequirement> dev_res_req :
                 arch_res_req->GetDeviceRequirementOptions()) {
              policy_->MapTask(task, *(dev_res_req->device()));
              std::cout << "\t\t[" << i << "]"
                        << dev_res_req->device()->get_name() << " -> "
                        << dev_res_req->res_req().get(MEMORY) << "B, VCU "
                        << dev_res_req->res_req().get(VCU) << "\n";
              ++i;
            }
          }
        }
      } else if (r->is_dev_req()) {
        DeviceRequirement *dev_res_req =
            dynamic_cast<DeviceRequirement *>(r.get());
        policy_->MapTask(task, *(dev_res_req->device()));
        std::cout << "[Device Requirement]\n";
        std::cout << dev_res_req->device()->get_name() << " -> "
                  << dev_res_req->res_req().get(MEMORY) << "B, VCU "
                  << dev_res_req->res_req().get(VCU) << "\n";
      } else if (r->is_arch_req()) {
        std::cout << "[Architecture Requirement]\n";
        ArchitectureRequirement *arch_res_req =
            dynamic_cast<ArchitectureRequirement *>(r.get());
        uint32_t i = 0;
        for (std::shared_ptr<DeviceRequirement> dev_res_req :
             arch_res_req->GetDeviceRequirementOptions()) {
          policy_->MapTask(task, *(dev_res_req->device()));
          std::cout << "\t[" << i << "]" << dev_res_req->device()->get_name()
                    << " -> " << dev_res_req->res_req().get(MEMORY) << "B, VCU "
                    << dev_res_req->res_req().get(VCU) << "\n";
          ++i;
        }
      }
    }*/

    // TODO(wlr): Testing
    // Assign two random devices to each task

    std::random_shuffle(devices.begin(), devices.end());
    ResourcePool_t sample;
    sample.set(MEMORY, 0);
    sample.set(VCU, 500);

    task->assigned_devices.push_back(devices[0]);
    task->assigned_devices.push_back(devices[1]);

    // TODO(wlr): Maybe use shared_ptr<ResourcePool_t> to pass from existing
    // res_req? Cannot be shared pools between devices. These are copies here.
    task->device_constraints.insert({devices[0]->get_global_id(), sample});
    task->device_constraints.insert({devices[1]->get_global_id(), sample});

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
  this->reservable_tasks.enqueue(task);
}

void MemoryReserver::enqueue(std::vector<InnerTask *> &tasks) {
  for (InnerTask *task : tasks) {
    this->enqueue(task);
  }
}

size_t MemoryReserver::get_count() {
  size_t count = this->reservable_tasks.size();
  return count;
}

bool MemoryReserver::check_resources(InnerTask *task) {
  bool status = true;
  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    status = device_pool.check_greater<ResourceCategory::PERSISTENT>(task_pool);

    if (!status) {
      break;
    }
  }
  return status;
}

void MemoryReserver::reserve_resources(InnerTask *task) {
  // TODO(wlr): Add runtime error check if resource failure

  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    device_pool.decrease<ResourceCategory::PERSISTENT>(task_pool);
  }
}

void MemoryReserver::run(SchedulerPhase *runtime_reserver) {
  NVTX_RANGE("MemoryReserver::run", NVTX_COLOR_LIGHT_GREEN)

  this->status.reset();

  // Only one thread can reserve memory at a time.
  // Useful for a multi-threaded scheduler. Not needed for a single-threaded.
  std::unique_lock<std::mutex> lock(this->mtx);

  // TODO:: Dummy implementation that just passes tasks through
  bool has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask *task = this->reservable_tasks.front();

    if (task == nullptr) {
      throw std::runtime_error("MemoryReserver::run: task is nullptr");
    }

    // Is there enough memory on the devices to schedule this task?
    bool can_reserve = this->check_resources(task);
    if (can_reserve) {
      this->reserve_resources(task);
      this->reservable_tasks.pop();
      this->reserved_tasks_buffer.push_back(task);
    } else {
      // TODO:(wlr) we need some break condition to allow the scheduler to
      // continue if not enough resources are available Hochan, do you
      // have any ideas? One failure per scheduler loop (written here) is
      // bad. Is one failure per device per scheduler loop better?
      break;
    }

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
  this->runnable_tasks.enqueue(task);
  // std::cout << "Ready tasks after enqueue: " <<
  // this->ready_tasks.atomic_size()
  //          << std::endl;
}

void RuntimeReserver::enqueue(std::vector<InnerTask *> &tasks) {
  for (auto task : tasks) {
    this->enqueue(task);
  }
}

size_t RuntimeReserver::get_count() {
  size_t count = this->runnable_tasks.size();
  return count;
}

bool RuntimeReserver::check_resources(InnerTask *task) {
  bool status = true;
  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    status =
        device_pool.check_greater<ResourceCategory::NON_PERSISTENT>(task_pool);

    if (!status) {
      break;
    }
  }
  return status;
}

void RuntimeReserver::reserve_resources(InnerTask *task) {
  // TODO(wlr): Add runtime error check if resource failure

  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    device_pool.decrease<ResourceCategory::NON_PERSISTENT>(task_pool);
  }
}

void RuntimeReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("RuntimeReserver::run", NVTX_COLOR_LIGHT_GREEN)

  Launcher *launcher = dynamic_cast<Launcher *>(next_phase);
  this->status.reset();

  // Only one thread can reserve runtime resources at a time.
  // Useful for a multi-threaded scheduler. Not needed for a single-threaded.
  std::unique_lock<std::mutex> lock(this->mtx);

  bool has_task = true;

  while (has_task) {
    has_task = this->get_count() > 0;

    if (has_task) {
      InnerTask *task = this->runnable_tasks.front();

      if (task == nullptr) {
        throw std::runtime_error("RuntimeReserver::run: task is nullptr");
      }

      bool has_resources = check_resources(task);

      if (has_resources) {
        bool has_thread = scheduler->workers.get_num_available_workers() > 0;

        if (has_thread) {
          InnerTask *task = this->runnable_tasks.pop();
          InnerWorker *worker = scheduler->workers.dequeue_worker();

          // Decrease Resources
          reserve_resources(task);

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
