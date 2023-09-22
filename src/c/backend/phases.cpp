#include "include/phases.hpp"
#include "include/device.hpp"
#include "include/parray.hpp"
#include "include/policy.hpp"
#include "include/profiling.hpp"
#include "include/rl_environment.hpp"
#include "include/rl_task_mapper.hpp"
#include "include/resource_requirements.hpp"
#include "include/resources.hpp"
#include "include/runtime.hpp"
#include <algorithm>
#include <initializer_list>
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

void Mapper::drain_parray_buffer() {

  while (unmapped_created_parrays.size() > 0) {

    auto parray_location_size = unmapped_created_parrays.front();

    auto parray = std::get<0>(parray_location_size);
    DevID_t dev_id = std::get<1>(parray_location_size);
    size_t size = std::get<2>(parray_location_size);

    // Get the device mapped memory pool
    Device *device = this->device_manager->get_device_by_global_id(dev_id);
    auto &mapped_pool = device->get_mapped_pool();

    // Note(@dialecticDolt): We do not throw a warning for mapped memory usage

    // Increase the mapped pool by the size of the parray
    mapped_pool.increase<Resource::Memory>(size);

    unmapped_created_parrays.pop_front();
  }
}

void Mapper::map_task(InnerTask *task, DeviceRequirementList &chosen_devices) {
  const auto &parray_list = task->parray_list;
  PArrayTracker *parray_tracker = this->parray_tracker;

  for (int local_device_idx = 0; local_device_idx < chosen_devices.size();
       ++local_device_idx) {

    auto chosen_device_requirements = chosen_devices[local_device_idx];
    Device *chosen_device = chosen_device_requirements->device();
    DevID_t global_dev_id = chosen_device->get_global_id();

    auto &mapped_pool = chosen_device->get_mapped_pool();
    auto &task_pool = chosen_device_requirements->res_req();

    task->add_assigned_device(chosen_device);

    task->device_constraints.insert(
        {chosen_device->get_global_id(), task_pool});

    auto &parray_access_list = parray_list[local_device_idx];

    int num_data_tasks = parray_access_list.size();
    this->atomic_incr_num_mapped_tasks_device(global_dev_id);
    this->atomic_incr_num_mapped_data_tasks_device(global_dev_id,
                                                   num_data_tasks);

    mapped_pool.increase(task_pool);

    for (int i = 0; i < parray_access_list.size(); ++i) {
      auto &parray_access = parray_access_list[i];
      size_t mapped_size = parray_tracker->do_log(global_dev_id, parray_access);

      mapped_pool.increase<Resource::Memory>(mapped_size);
    }
  }

  task->finalize_assigned_devices();
}

void Mapper::run(SchedulerPhase *next_phase) {

  NVTX_RANGE("Mapper::run", NVTX_COLOR_LIGHT_GREEN)

  MemoryReserver *memory_reserver = dynamic_cast<MemoryReserver *>(next_phase);

  // Comment(lhc): This is a non-critical region
  // Comment(wlr): Why is this a noncritical region?
  // Comment(lhc): Only one thread performs this function.
  // Comment(wlr): I don't think this is safe to always assume. If mapping is
  // expensive and takes a long time (for captured subgraphs) then it is
  // possible that the scheduler will be blocked on this function for a long
  // time. It is possible that another thread could be used to run this
  // while the scheduler continues in a future version.

  // Assumptions:
  // Scheduler maps a task to a device.
  // Scheduler does not reserve any resource at this phase.

  bool has_task = this->get_count() > 0;

<<<<<<< HEAD
  if (!has_task) {
    return;
  }

  if (this->scheduler->is_task_mapping_log_registered()) {
    bool attemptable{true};
    while (has_task &&
        this->scheduler->task_mapping_log.size() >
        this->scheduler->task_mapping_log_ptr && attemptable) {
      auto [task_name, device_id] =
          this->scheduler->task_mapping_log[
              this->scheduler->task_mapping_log_ptr];
      auto found = this->scheduler->task_name_to_task.find(task_name);
      if (found != this->scheduler->task_name_to_task.end()) {
        InnerTask* found_task = found->second;
        PlacementRequirementCollections &placement_req_options =
            found_task->get_placement_req_options();
        std::vector<std::shared_ptr<PlacementRequirementBase>>
            placement_req_options_vec =
                placement_req_options.get_placement_req_opts_ref();
        ParlaDevice *chosen_device{nullptr};
        std::shared_ptr<DeviceRequirement> chosen_device_req;
        if (placement_req_options_vec[0]->is_dev_req()) {
          chosen_device_req =
              std::dynamic_pointer_cast<DeviceRequirement>(
                  placement_req_options_vec[0]);
          chosen_device = chosen_device_req->device();
        } else {
          // XXX(hc): Assume that only single architecture requirement is in.
          ArchitectureRequirement *arch_req =
              dynamic_cast<ArchitectureRequirement*>(
                  placement_req_options_vec[0].get());
          auto placement_options = arch_req->GetDeviceRequirementOptions();
          // TODO(hc): placement_options is not sorted.
          for (size_t k = 0; k < placement_options.size(); ++k) {
            if (device_id == placement_options[k]->device()->get_global_id()) {
              chosen_device_req = placement_options[k];
              chosen_device = chosen_device_req->device();
              break;
            }
          }
        }
        DevID_t global_dev_id = chosen_device->get_global_id();
        found_task->assigned_devices.push_back(chosen_device);
        found_task->device_constraints.insert(
            {global_dev_id, chosen_device_req->res_req()});
        this->scheduler->task_mapping_log_ptr++;
        // This is a dummy pop and hack.
        // So if this size becomes 0, task mapping can be skipped.
        this->mappable_tasks.front_and_pop();
        this->mapped_tasks_buffer.push_back(found_task);
        has_task = this->get_count() > 0;
      } else {
        attemptable = false;
      }
    };
  } else {
    size_t num_task_mapping_attempt{0};
    while (has_task && num_task_mapping_attempt < 3) {
      InnerTask *task = this->mappable_tasks.front_and_pop();
      PlacementRequirementCollections &placement_req_options =
          task->get_placement_req_options();
      std::vector<std::shared_ptr<PlacementRequirementBase>>
          placement_req_options_vec =
              placement_req_options.get_placement_req_opts_ref();
      const std::vector<std::vector<std::pair<parray::InnerPArray*, AccessMode>>>
          &parray_list = task->parray_list;
      std::vector<std::shared_ptr<DeviceRequirement>> chosen_devices;
      policy_->run_task_mapping(task, this->scheduler, &chosen_devices,
          parray_list, &placement_req_options_vec);
      if (chosen_devices.empty()) {
        // It means that none of the devices is available for this task.
        // If it is, reenqueue the task to the mappable task queue.
        this->enqueue(task);
      } else {
        std::vector<std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
            *parray_list = &(task->parray_list);
        task->device_constraints.clear();
        for (size_t i = 0; i < chosen_devices.size(); ++i) {
          assert(chosen_devices[i] != nullptr);
          ParlaDevice *chosen_device = chosen_devices[i]->device();
          DevID_t global_dev_id = chosen_device->get_global_id();
          task->assigned_devices.push_back(chosen_device);
          task->device_constraints.insert(
              {chosen_device->get_global_id(), chosen_devices[i]->res_req()});
          task->mapping_time_epochs =
              this->scheduler->get_device_manager()->current_timepoint_count_from_beginning();
          this->scheduler->assign_task_mapping_id(task);
          // Increase the number of mapped tasks as the number of PArrays
          // since the corresponding data movement tasks will be created.
          this->scheduler->atomic_incr_num_mapped_tasks_device(global_dev_id);
          this->scheduler->atomic_incr_num_tasks_mapped_states(global_dev_id);

          LOG_INFO("Debug",
              "Mapping {} (Dev. {}): mapped {} res-reserved {} ready {} running {} total {}",
              task->name, global_dev_id,
              this->scheduler->
                  atomic_load_dev_num_tasks_mapped_states(global_dev_id),
              this->scheduler->
                  atomic_load_dev_num_tasks_resreserved_states(global_dev_id),
              this->scheduler->
                  atomic_load_dev_num_ready_tasks(global_dev_id),
              this->scheduler->
                  atomic_load_dev_num_running_tasks(global_dev_id),
              this->scheduler->
                  atomic_load_dev_num_mapped_tasks_device(global_dev_id));

          for (size_t j = 0; j < (*parray_list)[i].size(); ++j) {
            parray::InnerPArray *parray = (*parray_list)[i][j].first;
            this->scheduler->get_parray_tracker()->reserve_parray(*parray,
                                                                  chosen_device);
            parray->incr_num_active_tasks(global_dev_id);
          }
        }
#if 0
        std::cout << "chosen device size:" << chosen_devices.size() << "[done] \n" <<
          std::flush;

        std::cout << "[Mapper] Task name:" << task->get_name() << ", " << task
                  << "\n" << std::flush;
        for (size_t i = 0; i < task->assigned_devices.size(); ++i) {
          std::cout << "\t [" << i << "] " << ", " << task->assigned_devices[i]->get_global_id() <<
            ", "
                    << task->assigned_devices[i]->get_name() << "\n" << std::flush;
          auto res = task->device_constraints[task->assigned_devices[i]
                                                  ->get_global_id()];
          std::cout << "\t memory:" << res.get(Resource::Memory)
                    << ", vcu:" << res.get(Resource::VCU) << "\n" << std::flush;
        }
#endif
        this->mapped_tasks_buffer.push_back(task);
        this->scheduler->register_task_mapping_log(task);
      }
      has_task = this->get_count() > 0;
      num_task_mapping_attempt ++;
    } // while there are mappable tasks
  }
=======
  has_task = this->get_count() > 0;

  // In order to overlap scheduler phases and task execution,
  // use threshold of the number of tasks to be mapped.
  size_t num_task_mapping_attempt{0};

  this->drain_parray_buffer();

  // TODO Fix Issue #108
  while (has_task && num_task_mapping_attempt < 20) {

    this->drain_parray_buffer();

    // unmapped_created_parrays.lock();

    // Comment(wlr): this assumes the task is always able to be mapped.
    InnerTask *task = this->mappable_tasks.front_and_pop();

    PlacementRequirementCollections &placement_req_options =
        task->get_placement_req_options();
    PlacementRequirementList placement_req_options_vec =
        placement_req_options.get_placement_req_opts_ref();

    DeviceRequirementList chosen_devices;
    const auto &parray_list = task->parray_list;

    policy_->run_task_mapping(task, *this, &chosen_devices, parray_list,
                              &placement_req_options_vec);

    if (chosen_devices.empty()) {
      // No devices are available for this task.
      // Reenqueue the task to the mappable task queue to try again.
      this->enqueue(task);
    } else {
      this->map_task(task, chosen_devices);
      this->mapped_tasks_buffer.push_back(task);
    }

    // unmapped_created_parrays.unlock();
    has_task = this->get_count() > 0;
    ++num_task_mapping_attempt;
  } // while there are mappable tasks

>>>>>>> dev
  for (InnerTask *mapped_task : this->mapped_tasks_buffer) {
    mapped_task->notify_dependents(this->enqueue_buffer, TaskState::MAPPED);
    this->scheduler->enqueue_tasks(this->enqueue_buffer);
    this->enqueue_buffer.clear();

    if (this->scheduler->is_task_mapping_log_registered()) {
      mapped_task->num_unreserved_dependencies.store(1);
    }

    bool enqueue_flag =
        (mapped_task->num_unreserved_dependencies.fetch_sub(1) == 1);
    if (enqueue_flag) {
      mapped_task->set_status(TaskStatus::RESERVABLE);
      memory_reserver->enqueue(mapped_task);
    }
  }
  this->mapped_tasks_buffer.clear();
}

/**************************/
// Reserved Phase implementation

void MemoryReserver::drain_parray_buffer() {

  while (unreserved_created_parrays.size() > 0) {
    auto parray_location_size = unreserved_created_parrays.front();

    auto parray = std::get<0>(parray_location_size);
    DevID_t dev_id = std::get<1>(parray_location_size);
    size_t size = std::get<2>(parray_location_size);

    // Get the device mapped memory pool
    Device *device = this->device_manager->get_device_by_global_id(dev_id);
    auto &reserved_pool = device->get_reserved_pool();

    bool status = reserved_pool.check_greater<Resource::Memory>(size);
    if (!status) {
      std::cout
          << "WARNING: MemoryReserver::drain_parray_buffer: not enough memory"
          << std::endl;
      std::cout << "MemoryReserver::drain_parray_buffer: size: " << size
                << std::endl;
      std::cout << "Free memory: " << reserved_pool.get<Resource::Memory>()
                << std::endl;
    }

    // Increase the mapped pool by the size of the parray
    reserved_pool.decrease<Resource::Memory>(size);

    unreserved_created_parrays.pop_front();
  }
}

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
  for (ParlaDevice *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();
<<<<<<< HEAD
    status = device_pool.check_greater<ResourceCategory::Persistent>(task_pool);
=======

    status =
        device_pool.check_greater<Resource::PersistentResources>(task_pool);

>>>>>>> dev
    if (!status) {
      break;
    }
  }
  return status;
}

bool MemoryReserver::check_data_resources(InnerTask *task) {
  bool status = true;
  const auto &parray_list = task->parray_list;
  const auto &assigned_devices = task->assigned_devices;
  auto parray_tracker = this->parray_tracker;

  // Iterate through all PArray inputs
  for (DevID_t local_device_idx = 0; local_device_idx < assigned_devices.size();
       ++local_device_idx) {

    const auto &parray_access_list = parray_list[local_device_idx];

    // Get the device reserved memory pool
    Device *device = task->assigned_devices[local_device_idx];
    size_t size_on_device = 0;
    auto &reserved_pool = device->get_reserved_pool();

    for (int i = 0; i < parray_access_list.size(); ++i) {

      auto &parray_access = parray_access_list[i];
      InnerPArray *parray = parray_access.first;
      AccessMode access_mode = parray_access.second;

      parray->incr_num_referring_tasks(device->get_global_id());

      // If the PArray is not an input, then we don't need to check size
      // Note(@dialecticDolt):
      // There is literally no such thing as an out type in our syntax why do we
      // keep it around.
      if (access_mode == AccessMode::OUT) {
        continue;
      }

      // Get the expected additional size of the PArray on the device
      size_t size =
          parray_tracker->check_log(device->get_global_id(), parray_access);
      size_on_device += size;
    }

    // Check if the device has enough memory to store all PArray inputs
    bool device_status =
        reserved_pool.check_greater<Resource::Memory>(size_on_device);

    size_t necessary_free_bytes = size_on_device * 10;
    if (!reserved_pool.check_greater<Resource::Memory>(necessary_free_bytes)) {
      if (this->scheduler->get_mm_evictable_bytes(device->get_global_id())
          > size_on_device) {
        std::cout << "Eviction manager is invoked\n" << std::flush;
        // If a device has not enough memory, activate eviction manager
        this->scheduler->set_memory_size_to_evict(
            necessary_free_bytes, device->get_global_id());
        this->scheduler->break_for_eviction = true;
      }
    }

    status = status && device_status;
    if (!status) {
      break;
    }
  }

  return status;
}

void MemoryReserver::reserve_resources(InnerTask *task) {
  // TODO(wlr): Add runtime error check if resource failure

  for (ParlaDevice *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();
    device_pool.decrease<Resource::PersistentResources>(task_pool);
  }
}

void MemoryReserver::reserve_data_resources(InnerTask *task) {
  const auto &parray_list = task->parray_list;
  auto parray_tracker = this->parray_tracker;
  const auto &assigned_devices = task->assigned_devices;

  // Iterate through all PArray inputs
  for (DevID_t local_device_idx = 0; local_device_idx < assigned_devices.size();
       ++local_device_idx) {
    const auto &parray_access_list = parray_list[local_device_idx];

    // Get the device reserved memory pool
    Device *device = assigned_devices[local_device_idx];
    size_t size_on_device = 0;
    auto &reserved_pool = device->get_reserved_pool();

    for (int i = 0; i < parray_access_list.size(); ++i) {
      auto &parray_access = parray_access_list[i];
      InnerPArray *parray = parray_access.first;
      AccessMode access_mode = parray_access.second;

      // Register this PArray to eviction manager's table.
      // OUT also should be added to the table since
      // it will be created and use memory.
      this->scheduler->grab_parray_reference(parray, device->get_global_id());
      if (access_mode == AccessMode::OUT) {
        continue;
      }

      // Get the expected additional size of the PArray on the device
      size_t size =
          parray_tracker->do_log(device->get_global_id(), parray_access);
      size_on_device += size;
    }

    // Reserve the memory for all PArray inputs on the device
    reserved_pool.decrease<Resource::Memory>(size_on_device);
  }
}

void MemoryReserver::create_datamove_tasks(InnerTask *task) {
  // Get a list of the parrays the current task holds.
  const auto &parray_list = task->parray_list;
  const auto &assigned_devices = task->assigned_devices;

  std::string task_base_name = task->get_name();
  std::vector<InnerTask *> data_tasks;

  for (size_t local_device_idx = 0; local_device_idx < assigned_devices.size();
       ++local_device_idx) {

    auto device = assigned_devices[local_device_idx];
    auto &parray_access_list = parray_list[local_device_idx];

    for (size_t j = 0; j < parray_access_list.size(); ++j) {
      auto &parray_access = parray_access_list[j];
      InnerPArray *parray = parray_access.first;
      AccessMode access_mode = parray_access.second;

      // Create a data movement task for each PArray.
      // TODO(hc): id should be updated!
      std::string task_name = task_base_name + ".dm." + std::to_string(j);

      InnerDataTask *datamove_task = new InnerDataTask(
<<<<<<< HEAD
          // TODO(hc): id should be updated!
          task_base_name + ".dm." + std::to_string(i) + "-" +
          std::to_string(j), 0, parray, access_mode,
          i);
=======
          task_name, 0, parray, access_mode, local_device_idx);

>>>>>>> dev
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

        // NOTE(@dialecticDolt): Please fix this memory leak

        for (size_t t = 0; t < parray_task_list.size(); ++t) {
          if (parray_task_list.at(t)->id == parray_dependency->id) {
            data_task_dependencies.push_back(parray_dependency);
          }
        }
      }

      datamove_task->add_dependencies(data_task_dependencies, true);
      datamove_task->add_assigned_device(device);

      int copy_engines = 1;

      datamove_task->device_constraints.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(device->get_global_id()),
          std::forward_as_tuple(
              std::initializer_list<Resource_t>({0, 0, copy_engines})));

<<<<<<< HEAD
      data_tasks.push_back(datamove_task);
      // Data movement tasks should be registered at here;
      // these are not enqeueued through scheduler::enqueue_task().
      this->scheduler->task_name_to_task.emplace(datamove_task->name, datamove_task);
=======
>>>>>>> dev
      // Add the created data movement task to a reserved task queue.
      datamove_task->set_state(TaskState::RESERVED);
      this->scheduler->increase_num_active_tasks();
      this->reserved_tasks_buffer.push_back(datamove_task);

      data_tasks.push_back(datamove_task);
    }
  }

  // Create dependencies between the compute task and all its data tasks
  task->add_dependencies(data_tasks, true);
}

void MemoryReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("MemoryReserver::run", NVTX_COLOR_LIGHT_GREEN)

  RuntimeReserver *runtime_reserver =
      dynamic_cast<RuntimeReserver *>(next_phase);

  // Only one thread can reserve memory at a time.

  this->drain_parray_buffer();

  bool has_task = this->get_count() > 0;
  while (has_task) {

    this->drain_parray_buffer();

    unreserved_created_parrays.lock();

    InnerTask *task = this->reservable_tasks->front();

    if (task == nullptr) {
      unreserved_created_parrays.unlock();
      throw std::runtime_error("MemoryReserver::run: task is nullptr");
    }

    // Is there enough memory on the devices to schedule this task?
    // (internal resources)
    bool can_reserve = this->check_resources(task);

    // Is there enough memory on the devices to store this tasks inputs?
    // (parray resources)
    bool can_reserve_data = this->check_data_resources(task);

    if (can_reserve && can_reserve_data) {
      this->reserve_resources(task);
      this->reserve_data_resources(task);

      this->reservable_tasks->pop();
      this->create_datamove_tasks(task);
      this->reserved_tasks_buffer.push_back(task);

      // A task in this scope is only a computation task.
      for (ParlaDevice *device : task->assigned_devices) {
        DevID_t global_dev_id = device->get_global_id();
        this->scheduler->atomic_decr_num_tasks_mapped_states(global_dev_id);
        this->scheduler->atomic_incr_num_tasks_resreserved_states(global_dev_id);
        LOG_INFO("Debug",
            "ResReserved {} (Dev. {}): mapped {} res-reserved {} ready {} running {} total {}",
            task->name, global_dev_id,
            this->scheduler->
                atomic_load_dev_num_tasks_mapped_states(global_dev_id),
            this->scheduler->
                atomic_load_dev_num_tasks_resreserved_states(global_dev_id),
            this->scheduler->
                atomic_load_dev_num_ready_tasks(global_dev_id),
            this->scheduler->
                atomic_load_dev_num_running_tasks(global_dev_id),
            this->scheduler->
                atomic_load_dev_num_mapped_tasks_device(global_dev_id));
      }
    } else {
      // TODO:(wlr) we need some break condition to allow the scheduler to
      // continue if not enough resources are available Hochan, do you
      // have any ideas? One failure per scheduler loop (written here) is
      // bad. Is one failure per device per scheduler loop better?
      unreserved_created_parrays.unlock();
      break;
    }

    unreserved_created_parrays.unlock();
    has_task = this->get_count() > 0;
  }

  for (InnerTask *reserved_task : this->reserved_tasks_buffer) {
    reserved_task->notify_dependents(this->enqueue_buffer, TaskState::RESERVED);
    this->scheduler->enqueue_tasks(this->enqueue_buffer);
    this->enqueue_buffer.clear();

    // TODO:(wlr) Create and possibly enqueue data movement tasks

    // Possibly enqueue this task
    bool enqueue_flag =
        (reserved_task->num_blocking_dependencies.fetch_sub(1) == 1);

    if (enqueue_flag) {
      reserved_task->set_status(TaskStatus::RUNNABLE);
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
    this->runnable_tasks->enqueue(task);
  } else {
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
  for (ParlaDevice *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    status = device_pool.check_greater<Resources<Resource::VCU>>(task_pool);

    if (!status) {
      break;
    }
  }
  return status;
}

bool RuntimeReserver::check_data_resources(InnerTask *task) {
  bool status = true;
  for (ParlaDevice *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();

    status = device_pool.check_greater<Resource::MovementResources>(task_pool);

    if (!status) {
      break;
    }
  }
  return status;
}

void RuntimeReserver::reserve_resources(InnerTask *task) {
  // TODO(wlr): Add runtime error check if resource failure
  for (ParlaDevice *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();
    device_pool.decrease<Resource::NonPersistentResources>(task_pool);
  }
}

void RuntimeReserver::reserve_data_resources(InnerTask *task) {
  // TODO(wlr): Add runtime error check if resource failure
  for (ParlaDevice *device : task->assigned_devices) {
    ResourcePool_t &task_pool =
        task->device_constraints[device->get_global_id()];
    ResourcePool_t &device_pool = device->get_reserved_pool();
    device_pool.decrease<Resource::MovementResources>(task_pool);
  }
}

void RuntimeReserver::run(SchedulerPhase *next_phase) {
  NVTX_RANGE("RuntimeReserver::run", NVTX_COLOR_LIGHT_GREEN)

  Launcher *launcher = dynamic_cast<Launcher *>(next_phase);

  // Only one thread can reserve runtime resources at a time.
  // Useful for a multi-threaded scheduler. Not needed for a single-threaded.
  // std::unique_lock<std::mutex> lock(this->mtx);

  // Try to launch as many compute tasks as possible
  int fail_count = 0;
  int max_fail = this->runnable_tasks->get_num_devices() * 2;
  bool has_task = this->get_compute_count() > 0 ||
                  this->get_movement_count() > 0;
  int num_tasks = 0;
<<<<<<< HEAD
  if (this->scheduler->is_task_launching_log_registered()) {
    // If task launching order is registered,
    // follow that instead of getting a task from the queue.
    while (has_task &&
        this->scheduler->task_launching_log.size() >
        this->scheduler->task_launching_log_ptr) {
      std::string task_name =
          this->scheduler->task_launching_log[
              this->scheduler->task_launching_log_ptr];
      auto found = this->scheduler->task_name_to_task.find(task_name);
      if (found != this->scheduler->task_name_to_task.end()) {
        InnerTask* found_task = found->second;
        if (found_task->get_status() != Task::RUNNABLE) {
          // If task is not ready, break this loop.
          break;
=======
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
          // break;        // No more workers available
>>>>>>> dev
        }
        bool has_resources = check_resources(found_task);
        if (!has_resources) {
          // If resource is not ready, break this loop.
          break;
        }
        bool has_thread =
            this->scheduler->workers.get_num_available_workers() > 0;
        if (!has_thread) {
          // If thread is not ready, break this loop.
          break;
        }
        InnerWorker *worker = scheduler->workers.dequeue_worker();
        // Pop a single task from the queue; since launching replay mode 
        // only needs information about if there is remaining task or not.
        for (ParlaDevice *device : found_task->assigned_devices) {
          DevID_t global_dev_id = device->get_global_id();
          this->scheduler->atomic_decr_num_tasks_resreserved_states(
              global_dev_id);
          this->scheduler->atomic_incr_num_ready_tasks(global_dev_id);
        }
        this->scheduler->task_launching_log_ptr++;
        launcher->enqueue(found_task, worker);
        //std::cout << found_task->name << " next task:" <<
        //  this->scheduler->task_launching_log[
        //      this->scheduler->task_launching_log_ptr] << "\n" << std::flush;

<<<<<<< HEAD
        // To consider global_0 which can be run multiple times.
        // In this case, the launcher could run the old object instead of waiting
        // for the new object.
        this->scheduler->task_name_to_task.erase(task_name);
        this->status.increase(RuntimeReserverState::Success);
        if (found_task->is_data_task()) {
          this->movement_tasks->pop();
=======
  // Try to launch as many data movement tasks as possible
  has_task = true;
  num_tasks = 0;
  while (has_task) {
    num_tasks = this->get_movement_count();

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
>>>>>>> dev
        } else {
          this->runnable_tasks->pop();
        }
        has_task = this->get_compute_count() > 0 ||
                   this->get_movement_count() > 0;
      } else {
        break;
      }
    };
  } else {
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
            // A task in this scope is only a computation task.
            for (ParlaDevice *device : task->assigned_devices) {
              DevID_t global_dev_id = device->get_global_id();
              this->scheduler->atomic_decr_num_tasks_resreserved_states(global_dev_id);
              this->scheduler->atomic_incr_num_ready_tasks(global_dev_id);
              LOG_INFO("Debug",
                  "Ready {} (Dev. {}): mapped {} res-reserved {} ready {} running {} total {}",
                  task->name, global_dev_id,
                  this->scheduler->
                      atomic_load_dev_num_tasks_mapped_states(global_dev_id),
                  this->scheduler->
                      atomic_load_dev_num_tasks_resreserved_states(global_dev_id),
                  this->scheduler->
                      atomic_load_dev_num_ready_tasks(global_dev_id),
                  this->scheduler->
                      atomic_load_dev_num_running_tasks(global_dev_id),
                  this->scheduler->
                      atomic_load_dev_num_mapped_tasks_device(global_dev_id));
            }
            launcher->enqueue(task, worker);
            this->status.increase(RuntimeReserverState::Success);
          } else {
            this->status.increase(RuntimeReserverState::NoWorker);
            fail_count++; // += max_fail;
            // break;        // No more workers available
          }
        } else {
          this->status.increase(RuntimeReserverState::NoResource);
          fail_count++;
          // break; // No more resources available
        }
      } else {
        this->status.increase(RuntimeReserverState::NoTask);
        fail_count++; //+= max_fail;
        // break;        // No more tasks available
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
}

/**************************/
// Launcher Phase implementation

void Launcher::enqueue(InnerTask *task, InnerWorker *worker) {
  NVTX_RANGE("Launcher::enqueue", NVTX_COLOR_LIGHT_GREEN)

  // Immediately launch task
  task->set_state(TaskState::RUNNING);
  this->num_running_tasks++;

  for (size_t i = 0; i < task->assigned_devices.size(); ++i) {
    ParlaDevice *device = task->assigned_devices[i];
    DevID_t global_dev_id = device->get_global_id();
    device->end_device_idle();
    if (!task->is_data_task()) {
      this->scheduler->atomic_decr_num_ready_tasks(global_dev_id);
      this->scheduler->atomic_incr_num_running_tasks(global_dev_id);
      LOG_INFO("Debug",
          "Running {} (Dev. {}): mapped {} res-reserved {} ready {} running {} total {}",
          task->name, global_dev_id,
          this->scheduler->
              atomic_load_dev_num_tasks_mapped_states(global_dev_id),
          this->scheduler->
              atomic_load_dev_num_tasks_resreserved_states(global_dev_id),
          this->scheduler->
              atomic_load_dev_num_ready_tasks(global_dev_id),
          this->scheduler->
              atomic_load_dev_num_running_tasks(global_dev_id),
          this->scheduler->
              atomic_load_dev_num_mapped_tasks_device(global_dev_id));
    }
    //std::cout << task->name << " <- " << device->get_global_id() << "\n";
  }

  this->scheduler->assign_task_launching_id(task);

  // Log launching task order.
  if (!this->scheduler->is_task_launching_log_registered()) {
    if (this->scheduler->task_log_register_counter.load() == 1) {
      // Only register the second iteration; this is because it is possible
      // that there are tasks spawned from the second iteration like
      // Reset in Cholesky.
      this->scheduler->register_task_launching_log(task);
    }
  }

  if (task->name.find("end_task_graph") != std::string::npos) {
    TimePoint now = std::chrono::system_clock::now();
    TimePoint initial_time_epoch = this->scheduler->get_initial_epoch();
    this->scheduler->epoch_end_epochs =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - initial_time_epoch).count();
    double exec_time_ms =
        this->scheduler->epoch_end_epochs - this->scheduler->epoch_begin_epochs;
    std::cout << "Exec. time on scheduler side:" <<
        this->scheduler->epoch_end_epochs - this->scheduler->epoch_begin_epochs << "\n";

    if (dynamic_cast<RLTaskMappingPolicy*>(this->
            scheduler->mapper->get_policy_raw_pointer()) != nullptr) {
      this->scheduler->mapper->get_policy_raw_pointer()->
          evaluate_current_epoch(
              exec_time_ms, this->scheduler->previous_exec_time);
    }
    this->scheduler->end_task_graph_is_last = true;
    this->scheduler->previous_exec_time =
        std::min(this->scheduler->previous_exec_time, exec_time_ms);
  }
  // We need one more condition check since Parla awaitness
  // spawns the parent task when awaitness on the last task
  // is done.
  else if (this->scheduler->end_task_graph_is_last) {
    this->scheduler->end_task_graph_is_last = false;
    this->scheduler->reset_task_mapping_id();
    this->scheduler->reset_task_launching_id();

    this->scheduler->reset_task_launching_log_pointer();
    this->scheduler->reset_task_mapping_log_pointer();
    this->scheduler->reset_task_name_to_task();

    // global_0 after the last task will be here.
    // so this should be also skipped.
    //this->scheduler->task_launching_log_ptr++;

    if (!this->scheduler->is_task_mapping_log_registered()) {
      std::cout << "task mapping order:\n";
      for (size_t i = 0; i < this->scheduler->task_mapping_log.size(); ++i) {
        std::cout << i << ", " << this->scheduler->task_mapping_log[i].first <<
          ", " << this->scheduler->task_mapping_log[i].second << "\n" << std::flush;
      }
    }
    if (!this->scheduler->is_task_launching_log_registered()) {
      std::cout << "task launching order:\n";
      for (size_t i = 0; i < this->scheduler->task_launching_log.size(); ++i) {
        std::cout << i << ", " << this->scheduler->task_launching_log[i] << "\n" << std::flush;
      }
    }

    this->scheduler->complete_task_order_logs(task);
    ++this->scheduler->num_epochs;
  }

  // Assign task to thread and notify via c++ condition variable.
  // No GIL needed until worker wakes.
  worker->assign_task(task);
<<<<<<< HEAD
=======

>>>>>>> dev
  LOG_INFO(WORKER, "Assigned {} to {}", task, worker);
}

void Launcher::run() {
  throw std::runtime_error("Launcher::run() not implemented.");
}

