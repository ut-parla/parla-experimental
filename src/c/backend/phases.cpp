#include "include/phases.hpp"
#include "include/device.hpp"
#include "include/parray.hpp"
#include "include/policy.hpp"
#include "include/profiling.hpp"
#include "include/resource_requirements.hpp"
#include "include/resources.hpp"
#include "include/runtime.hpp"
#include <algorithm>
#include <random>
#include <utility>

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

void Mapper::run(SchedulerPhase *next_phase) {

  NVTX_RANGE("Mapper::run", NVTX_COLOR_LIGHT_GREEN)

  //std::cout << "Mapper::run" << std::endl;

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
    //Comment(wlr): this assumes the task is always able to be mapped.
    InnerTask *task = this->mappable_tasks.front_and_pop();
    PlacementRequirementCollections &placement_req_options =
        task->get_placement_req_options();
    std::vector<std::shared_ptr<PlacementRequirementBase>>
        placement_req_options_vec =
            placement_req_options.get_placement_req_opts_ref();
    std::vector<std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
        &parray_list = task->parray_list;
    // A set of chosen devices to a task.
    Score_t best_score{-1};
    std::vector<std::shared_ptr<DeviceRequirement>> chosen_devices;

    // Iterate all placement requirements passed by users and calculate
    // scores based on a policy.
    for (std::shared_ptr<PlacementRequirementBase> base_req :
         placement_req_options_vec) {
      if (base_req->is_multidev_req()) {
        // Multi-device placement requirements.
        // std::cout << "[Multi-device requirement]\n";
        MultiDeviceRequirements *mdev_reqs =
            dynamic_cast<MultiDeviceRequirements *>(base_req.get());
        std::vector<std::shared_ptr<DeviceRequirement>> mdev_reqs_vec;
        Score_t score{0};
        bool is_req_available = policy_->calc_score_mdevplacement(
            task, mdev_reqs, *this, &mdev_reqs_vec, &score, parray_list);
        if (!is_req_available) {
          continue;
        }
        if (best_score <= score) {
          best_score = score;
          chosen_devices.swap(mdev_reqs_vec);
        }
      } else if (base_req->is_dev_req()) {
        // A single device placement requirement.
        std::shared_ptr<DeviceRequirement> dev_req =
            std::dynamic_pointer_cast<DeviceRequirement>(base_req);
        Score_t score{0};
        bool is_req_available = policy_->calc_score_devplacement(
            task, dev_req, *this, &score, parray_list[0]);
        if (!is_req_available) {
          continue;
        }
        if (best_score <= score) {
          assert(dev_req != nullptr);
          best_score = score;
          chosen_devices.clear();
          chosen_devices.emplace_back(dev_req);
        }
      } else if (base_req->is_arch_req()) {
        // A single architecture placement requirement.
        ArchitectureRequirement *arch_req =
            dynamic_cast<ArchitectureRequirement *>(base_req.get());
        std::shared_ptr<DeviceRequirement> chosen_dev_req{nullptr};
        Score_t chosen_dev_score{0};
        bool is_req_available = policy_->calc_score_archplacement(
            task, arch_req, *this, chosen_dev_req, &chosen_dev_score,
            parray_list[0]);
        if (!is_req_available) {
          continue;
        }
        if (best_score <= chosen_dev_score) {
          assert(chosen_dev_req != nullptr);
          best_score = chosen_dev_score;
          chosen_devices.clear();
          chosen_devices.emplace_back(chosen_dev_req);
        }
      }
    }


    if (chosen_devices.empty()) {
      // It means that none of the devices is available for this task.
      // If it is, re-enqueue the task to the mappable task queue.
      this->enqueue(task);
      //std::cout << "Task has not been mapped" << std::endl;
    } else {
      std::vector<std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          *parray_list = &(task->parray_list);
      for (size_t i = 0; i < chosen_devices.size(); ++i) {
        assert(chosen_devices[i] != nullptr);
        Device *chosen_device = chosen_devices[i]->device();
        DevID_t global_dev_id = chosen_device->get_global_id();
        task->assigned_devices.push_back(chosen_device);
        task->device_constraints.insert(
            {chosen_device->get_global_id(), chosen_devices[i]->res_req()});
        this->atomic_incr_num_mapped_tasks_device(
            chosen_device->get_global_id());
        for (size_t j = 0; j < (*parray_list)[i].size(); ++j) {
          parray::InnerPArray *parray = (*parray_list)[i][j].first;
          this->scheduler->get_parray_tracker()->reserve_parray(*parray,
                                                                chosen_device);
          parray->incr_num_active_tasks(global_dev_id);
        }
      }

      // std::cout << "[Mapper] Task name:" << task->get_name() << ", " << task
      //           << "\n";
      // for (size_t i = 0; i < task->assigned_devices.size(); ++i) {
      //   std::cout << "\t [" << i << "] "
      //             << task->assigned_devices[i]->get_name() << "\n";
      //   auto res = task->device_constraints[task->assigned_devices[i]
      //                                           ->get_global_id()];
      //   std::cout << "\t memory:" << res.get(Resource::Memory)
      //             << ", vcu:" << res.get(Resource::VCU) << "\n";
      // }
      //
      //std::cout << "task has been mapped!" << std::endl;

      this->mapped_tasks_buffer.push_back(task);
      this->atomic_incr_num_mapped_tasks();
    }
    has_task = this->get_count() > 0;
  } // while there are mappable tasks

  for (InnerTask *mapped_task : this->mapped_tasks_buffer) {
    std::cout << "notifying dependencies of mapped status" << std::endl;
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

void MemoryReserver::create_datamove_tasks(InnerTask *task) {
  // Get a list of the parrays the current task holds.
  const std::vector<std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
      &parray_list = task->parray_list;
  std::string task_base_name = task->get_name();
  std::vector<InnerTask *> data_tasks;
  data_tasks.reserve(parray_list.size());

  for (size_t i = 0; i < parray_list.size(); ++i) {
    for (size_t j = 0; j < parray_list[i].size(); ++j) {
      // Create a data movement task for each PArray.
      parray::InnerPArray *parray = parray_list[i][j].first;
      AccessMode access_mode = parray_list[i][j].second;
      InnerDataTask *datamove_task = new InnerDataTask(
          // TODO(hc): id should be updated!
          task_base_name + ".dm." + std::to_string(i), 0, parray, access_mode,
          i);
      auto &parray_task_list = parray->get_parent_parray()->get_task_list_ref();
      // Find dependency intersection between compute and data movement tasks.

      // TODO(hc): This is not the complete implementation.
      //           We will use a concurrent map for parray's
      //           task list as an optimization.

      std::vector<void *> compute_task_dependencies = task->get_dependencies();
      std::vector<InnerTask *> data_task_dependencies;
      for (size_t k = 0; k < compute_task_dependencies.size(); ++k) {
        InnerTask *parray_dependency =
            static_cast<InnerTask *>(compute_task_dependencies[k]);
        // The task list in PArray is currently thread safe since
        // we do not remove tasks from the list but just keep even completed
        // task as its implementation is easier.
        for (size_t t = 0; t < parray_task_list.size_unsafe(); ++t) {
          if (parray_task_list.at_unsafe(t)->id == parray_dependency->id) {
            data_task_dependencies.push_back(parray_dependency);
          }
        }
      }

      // TODO(hc): pass false to add_dependencies() as optimization.
      datamove_task->add_dependencies(data_task_dependencies, true);
      // Copy assigned devices to a compute task to a data movement task.
      // TODO(hc): When we support xpy, it should be devices corresponding
      //           to placements of the local partition.
      auto device = task->get_assigned_devices()[i];
      datamove_task->add_assigned_device(device);

      datamove_task->device_constraints.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(device->get_global_id()),
          std::forward_as_tuple(0, 0, 1));

      data_tasks.push_back(datamove_task);
      // Add the created data movement task to a reserved task queue.
      this->scheduler->increase_num_active_tasks();
      this->reserved_tasks_buffer.push_back(datamove_task);
    }
  }

  // Create dependencies between data move task and compute tasks.
  task->add_dependencies(data_tasks, true);
}

void MemoryReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("MemoryReserver::run", NVTX_COLOR_LIGHT_GREEN)

  //std::cout << "MemoryReserver::run" << std::endl;

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
      this->create_datamove_tasks(task);
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
  bool is_data_task = task->is_data_task();
  if (!is_data_task) {
    // std::cout << "RuntimeReserver::enqueue: compute task" << std::endl;
    this->runnable_tasks->enqueue(task);
  } else {
    // std::cout << "RuntimeReserver::enqueue: data task" << std::endl;
    this->movement_tasks->enqueue(task);
  }
}

void RuntimeReserver::enqueue(std::vector<InnerTask *> &tasks) {
  for (InnerTask *task : tasks) {
    this->enqueue(task);
  }
}

size_t RuntimeReserver::get_compute_count() {
  size_t count = this->runnable_tasks->size();
  return count;
}

size_t RuntimeReserver::get_movement_count() {
  size_t count = this->movement_tasks->size();
  return count;
}

size_t RuntimeReserver::get_count() {
  size_t count = this->get_compute_count() + this->get_movement_count();
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

bool RuntimeReserver::check_data_resources(InnerTask *task) {
  bool status = true;
  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    status = device_pool.check_greater<ResourceCategory::Movement>(task_pool);

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

void RuntimeReserver::reserve_data_resources(InnerTask *task) {
  // TODO(wlr): Add runtime error check if resource failure
  for (Device *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();
    device_pool.decrease<ResourceCategory::Movement>(task_pool);
  }
}

void RuntimeReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("RuntimeReserver::run", NVTX_COLOR_LIGHT_GREEN)

  //std::cout << "RuntimeReserver::run" << std::endl;

  Launcher *launcher = dynamic_cast<Launcher *>(next_phase);

  // Only one thread can reserve runtime resources at a time.
  // Useful for a multi-threaded scheduler. Not needed for a single-threaded.
  // std::unique_lock<std::mutex> lock(this->mtx);

  // Try to launch as many compute tasks as possible
  int fail_count = 0;
  int max_fail = this->runnable_tasks->get_num_devices() * 2;
  bool has_task = true;
  int num_tasks = 0;
  while (has_task && (fail_count < max_fail)) {
    num_tasks = this->get_compute_count();
    has_task = num_tasks > 0;
    if (has_task) {
      InnerTask *task = this->runnable_tasks->front();
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
          fail_count++; // += max_fail;
          //break;        // No more workers available
        }
      } else {
        this->status.increase(RuntimeReserverState::NoResource);
        fail_count++;
        //break; // No more resources available
      }
    } else {
      this->status.increase(RuntimeReserverState::NoTask);
      fail_count++; //+= max_fail;
      //break;        // No more tasks available
    }
  }

  // Try to launch as many data movement tasks as possible
  has_task = true;
  num_tasks = 0;
  while (has_task) {
    num_tasks = this->get_movement_count();
    // std::cout << "RuntimeReserver::run: num movement tasks: " << num_tasks
    //           << std::endl;
    has_task = num_tasks > 0;
    if (has_task) {
      InnerTask *task = this->movement_tasks->front();
      bool has_resources = check_data_resources(task);
      if (has_resources) {
        bool has_thread = scheduler->workers.get_num_available_workers() > 0;
        if (has_thread) {
          InnerTask *task = this->movement_tasks->pop();
          InnerWorker *worker = scheduler->workers.dequeue_worker();
          // Decrease Resources
          this->reserve_data_resources(task);
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

  //std::cout << "Launcher::enqueue" << std::endl;

  // Immediately launch task
  task->set_state(Task::RUNNING);
  this->num_running_tasks++;

  // Assign task to thread and notify via c++ condition variable.
  // No GIL needed until worker wakes.
  worker->assign_task(task);

  // std::cout << "Assigned " << task->name << " to " << worker->thread_idx
  //          << std::endl;
  LOG_INFO(WORKER, "Assigned {} to {}", task, worker);
}

void Launcher::run() {
  throw std::runtime_error("Launcher::run() not implemented.");
}
