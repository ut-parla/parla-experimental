#include "include/phases.hpp"
#include "include/device.hpp"
#include "include/policy.hpp"
#include "include/profiling.hpp"
#include "include/resource_requirements.hpp"
#include "include/resources.hpp"
#include "include/runtime.hpp"
#include <algorithm>
#include <random>

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

  MemoryReserver *memory_reserver = dynamic_cast<MemoryReserver *>(next_phase);

  // TODO: Refactor this so its readable without as many nested conditionals

  // This is a non-critical region
  // Comment(wlr): Why is this a noncritical region?

  // Assumptions:
  // Scheduler maps a task to a device.
  // Scheduler does not reserve any resource at this phase.

  // TODO(hc): for now, I'm planning task mapping without policy.

  bool has_task = true;

  has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask *task = this->mappable_tasks.front_and_pop();
#if 0

    // TODO(wlr): Testing
    // Assign a random device set to each task
    // TODO(wlr): This is just used for random verification
    // testing/benchmarking.
    //            Remove this when we implement a policy.

    // Just grabbing CPU VCUs for now.
    // Assume it exists and is first.

    auto valid_options = task->GetResourceRequirements();
    auto dev_res_reqs = valid_options.GetDeviceRequirementOptions();
    auto cpu_req = dev_res_reqs[0];

    ArchitectureRequirement *arch_req =
        dynamic_cast<ArchitectureRequirement *>(cpu_req.get());

    auto specific_device_req = arch_req->GetDeviceRequirementOptions()[0];
    int vcu = specific_device_req->res_req().get(Resource::VCU);

    // std::cout << "VCU: " << vcu << std::endl;

    std::vector<Device *> devices;
    devices.insert(devices.end(),
                   this->device_manager->get_devices(DeviceType::All).begin(),
                   this->device_manager->get_devices(DeviceType::All).end());

    std::random_device rd;
    std::mt19937 g(rd());
    // std::shuffle(devices.begin(), devices.end(), g);

    ResourcePool_t sample;
    sample.set(Resource::Memory, 0);
    sample.set(Resource::VCU, vcu);

    task->assigned_devices.push_back(devices[0]);
    // task->assigned_devices.push_back(devices[1]);

    // TODO(wlr): Maybe use shared_ptr<ResourcePool_t> to pass from existing
    // res_req? Cannot be shared pools between devices. These are copies here.
    task->device_constraints.insert({devices[0]->get_global_id(), sample});
    // task->device_constraints.insert({devices[1]->get_global_id(), sample});

    // std::cout << "Mapping task " << task->get_name() << " to devices "
    //           << devices[0]->get_name() << " and " << devices[1]->get_name()
    //           << std::endl;
#endif
    ResourceRequirementCollections &res_reqs = task->get_placement_req_options();
    std::vector<std::shared_ptr<DeviceRequirementBase>> dev_res_reqs =
        res_reqs.GetDeviceRequirementOptions();
    // A set of chosen devices to a task.
    Score_t best_score{0};
    std::vector<Device*> chosen_devices;
    for (std::shared_ptr<DeviceRequirementBase> base_res_req : dev_res_reqs) {
      if (base_res_req->is_multidev_req()) {
        // Multi-device placement requirements.
        std::cout << "[Multi-device requirement]\n";
        MultiDeviceRequirements *mdev_reqs =
            dynamic_cast<MultiDeviceRequirements *>(base_res_req.get());
        auto [score, dev_vec] =
            policy_->calc_score_mdevplacement(task, mdev_reqs, 
                this->atomic_load_total_num_mapped_tasks());
        if (best_score < score) {
          best_score = score;
          chosen_devices.swap(dev_vec);
        }
      } else if (base_res_req->is_dev_req()) {
        // A single device placement requirement.
        DeviceRequirement *dev_req =
            dynamic_cast<DeviceRequirement *>(base_res_req.get());
        auto [score, dev] =
          policy_->calc_score_devplacement(task, dev_req,
              this->atomic_load_total_num_mapped_tasks());
        if (best_score < score) {
          best_score = score;
          chosen_devices.clear();
          chosen_devices.emplace_back(dev);
        }
      } else if (base_res_req->is_arch_req()) {
        // A single architecture placement requirement.
        ArchitectureRequirement* arch_req =
            dynamic_cast<ArchitectureRequirement *>(base_res_req.get());
        auto [score, dev] =
          policy_->calc_score_archplacement(task, arch_req,
              this->atomic_load_total_num_mapped_tasks());
        if (best_score < score) {
          best_score = score;
          chosen_devices.clear();
          chosen_devices.emplace_back(dev);
        }
      }
    }

    this->mapped_tasks_buffer.push_back(task);
    // TODO(hc): this->atomic_incr_num_mapped_tasks(device id);
    this->atomic_incr_num_mapped_tasks(0);
    //this->device_manager->IncrAtomicTotalNumMappedTasks();
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
  this->reservable_tasks->enqueue(task);
}

void MemoryReserver::enqueue(std::vector<InnerTask *> &tasks) {
  for (InnerTask *task : tasks) {
    this->enqueue(task);
  }
}

size_t MemoryReserver::get_count() {
  size_t count = this->reservable_tasks->size();
  return count;
}

bool MemoryReserver::check_resources(InnerTask *task) {
  bool status = true;
  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    status = device_pool.check_greater<ResourceCategory::Persistent>(task_pool);

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

    device_pool.decrease<ResourceCategory::Persistent>(task_pool);
  }
}

void MemoryReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("MemoryReserver::run", NVTX_COLOR_LIGHT_GREEN)

  RuntimeReserver *runtime_reserver =
      dynamic_cast<RuntimeReserver *>(next_phase);

  // Only one thread can reserve memory at a time.
  // Useful for a multi-threaded scheduler. Not needed for a single-threaded.
  // std::unique_lock<std::mutex> lock(this->mtx);

  // TODO:: Dummy implementation that just passes tasks through
  bool has_task = this->get_count() > 0;
  while (has_task) {
    InnerTask *task = this->reservable_tasks->front();

    if (task == nullptr) {
      throw std::runtime_error("MemoryReserver::run: task is nullptr");
    }

    // Is there enough memory on the devices to schedule this task?
    bool can_reserve = this->check_resources(task);
    if (can_reserve) {
      this->reserve_resources(task);
      this->reservable_tasks->pop();
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
  this->runnable_tasks->enqueue(task);
}

void RuntimeReserver::enqueue(std::vector<InnerTask *> &tasks) {
  for (InnerTask *task : tasks) {
    this->enqueue(task);
  }
}

size_t RuntimeReserver::get_count() {
  size_t count = this->runnable_tasks->size();
  return count;
}

bool RuntimeReserver::check_resources(InnerTask *task) {
  bool status = true;
  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    status =
        device_pool.check_greater<ResourceCategory::NonPersistent>(task_pool);

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

    device_pool.decrease<ResourceCategory::NonPersistent>(task_pool);
  }
}

void RuntimeReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("RuntimeReserver::run", NVTX_COLOR_LIGHT_GREEN)

  Launcher *launcher = dynamic_cast<Launcher *>(next_phase);

  // Only one thread can reserve runtime resources at a time.
  // Useful for a multi-threaded scheduler. Not needed for a single-threaded.
  // std::unique_lock<std::mutex> lock(this->mtx);

  bool has_task = true;
  int num_tasks = 0;
  while (has_task) {
    num_tasks = this->get_count();
    has_task = num_tasks > 0;

    // std::cout << "RuntimeReserver::run: num_tasks = " << num_tasks <<
    // std::endl;
    if (has_task) {

      InnerTask *task = this->runnable_tasks->front();

      if (task == nullptr) {
        throw std::runtime_error("RuntimeReserver::run: task is nullptr");
      }

      bool has_resources = check_resources(task);

      if (has_resources) {
        bool has_thread = scheduler->workers.get_num_available_workers() > 0;

        if (has_thread) {
          InnerTask *task = this->runnable_tasks->pop();
          InnerWorker *worker = scheduler->workers.dequeue_worker();

          // Decrease Resources
          this->reserve_resources(task);

          launcher->enqueue(task, worker);

          this->status.increase(RuntimeReserverState::Success);
        } else {
          this->status.increase(RuntimeReserverState::NoWorker);
          break; // No more workers available
        }
      } else {
        this->status.increase(RuntimeReserverState::NoResource);
        break; // No more resources available
      }
    } else {
      this->status.increase(RuntimeReserverState::NoTask);
      break; // No more tasks available
    }
  }
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
