#include "include/policy.hpp"
#include "include/phases.hpp"

bool LocalityLoadBalancingMappingPolicy::calc_score_devplacement(
    InnerTask *task,
    const std::shared_ptr<DeviceRequirement> &dev_placement_req,
    const Mapper &mapper, Score_t *score,
    const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
              &parray_list) {
  const Device &device = *(dev_placement_req->device());
  DevID_t global_dev_id = device.get_global_id();
  //std::cout << "[Locality-aware- and Load-balancing mapping policy]\n";

  // Check device resource availability.
  if (!device.check_resource_availability(dev_placement_req.get())) {
    //std::cout << "Device resource failure!" << std::endl;
    return false;
  }

  // PArray locality.
  Score_t local_data{0}, nonlocal_data{0};
  for (size_t i = 0; i < parray_list.size(); ++i) {
    InnerPArray *parray = parray_list[i].first;
    if (parray_tracker_->get_parray_state(global_dev_id, parray->parent_id)) {
      local_data += parray->get_size();
    } else {
      nonlocal_data += parray->get_size();
    }
  }
  Resource_t device_memory_size = device.query_resource(Resource::Memory);
  local_data /= device_memory_size;
  nonlocal_data /= device_memory_size;

#if 0
  // TODO(hc): This metric is duplicated with data locality.
  // Check how many dependencies are mapped to the device candidate
  // being checked.
  // Note that this only considers dependencies specified in a task
  // @spawn decorator. So to speak, it considers immediate dependencies
  // and does not consider the whole dependency subtree.
  TaskList &dependencies = task->dependencies;
  size_t num_dependencies_on_device{0};
  for (size_t i = 0; i < dependencies.size_unsafe(); ++i) {
    InnerTask *dependency = dependencies.at_unsafe(i);
    for (Device *dependency_devices : dependency->assigned_devices) {
      if (dependency_devices->get_global_id() == device.get_global_id()) {
        ++num_dependencies_on_device;
        break;
      }
    }
  }
  // std::cout << num_dependencies_on_device << " of the " << task->get_name()
  //           << "s dependencies have been mapped to device " <<
  //           device.get_name()
  //           << "\n";
#endif

  // TODO(hc): memory load; but let me postpone this implementation because
  //           it may require nested for loops.

  // Calculate device load balancing.
  size_t total_num_mapped_tasks = mapper.atomic_load_total_num_mapped_tasks();
  size_t num_tasks_to_device =
      mapper.atomic_load_dev_num_mapped_tasks_device(device.get_global_id());
  double normalizd_device_load{0};
  if (total_num_mapped_tasks != 0) {
    normalizd_device_load =
        num_tasks_to_device / double(total_num_mapped_tasks);
  }

  // Avoid negative score and make this focus on load balancing if data
  // is not used.
  *score = 50;
  *score += (30.0 * local_data - 30.0 * nonlocal_data - 10 * normalizd_device_load);

  /*
  std::cout << "Device " << device.get_name() << "'s score: " << *score <<
    " for task "<< task->get_name() << " local data: " << local_data <<
    " non local data:" << nonlocal_data << " normalized device load:" <<
    normalizd_device_load << "\n";
  std::cout << "\t[Device Requirement in device Requirement]\n"
            << "\t\t" << dev_placement_req->device()->get_name() << " -> "
            << dev_placement_req->res_req().get(Resource::Memory) << "B, VCU"
            << dev_placement_req->res_req().get(Resource::VCU) << "\n";
  */
  return true;
}

bool LocalityLoadBalancingMappingPolicy::calc_score_archplacement(
    InnerTask *task, ArchitectureRequirement *arch_placement_req,
    const Mapper &mapper, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
    Score_t *chosen_dev_score,
    const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
            &parray_list, std::vector<bool> *is_dev_assigned) {
  Score_t best_score{0};
  std::shared_ptr<DeviceRequirement> best_device_req{nullptr};
  uint32_t i{0};
  bool is_arch_available{false};
  // For now the architecture placement has one resource requirement
  // regardless of the devices of the architecture. In the future,
  // we will allow a separate placement for each device.
  for (std::shared_ptr<DeviceRequirement> dev_req :
       arch_placement_req->GetDeviceRequirementOptions()) {
    Score_t score{0};
    DevID_t dev_global_id = dev_req->device()->get_global_id();
    if (is_dev_assigned != nullptr &&
            (*is_dev_assigned)[dev_global_id] == true) {
      // If this architecture placement is the member of a
      // multi-device task and this visiting device is already chosen
      // as one of the placements, skip it.
      continue;
    }
    bool is_dev_available =
        this->calc_score_devplacement(task, dev_req, mapper, &score,
            parray_list);
    if (!is_dev_available) {
      continue;
    }
    is_arch_available = true;
    if (best_score <= score) {
      best_score = score;
      best_device_req = dev_req;
    }
    ++i;
  }
  assert(best_device_req != nullptr);
  chosen_dev_req = best_device_req;
  *chosen_dev_score = best_score;
  return is_arch_available;
}

bool LocalityLoadBalancingMappingPolicy::calc_score_mdevplacement(
    InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
    const Mapper &mapper,
    std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
    Score_t *average_score,
    const std::vector<
        std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list) {
  *average_score = 0;
  const std::vector<std::shared_ptr<SinglePlacementRequirementBase>>
      &placement_reqs_vec = mdev_placement_req->get_placement_reqs_ref();
  member_device_reqs->resize(placement_reqs_vec.size());
  // Task mapper does not allow to map a multi-device task to the same device
  // multiple times. This vector marks an assigned device and filter it
  // out at the next device decision.
  std::vector<bool> is_dev_assigned(
      this->device_manager_->get_num_devices<DeviceType::All>(), false);
  // Iterate requirements of the devices specified in multi-device placement.
  // All of the member devices should be available.
  for (DevID_t did = 0; did < placement_reqs_vec.size(); ++did) {
    std::shared_ptr<SinglePlacementRequirementBase> placement_req =
        placement_reqs_vec[did];
    std::shared_ptr<DeviceRequirement> dev_req{nullptr};
    Score_t score{0};
    bool is_member_device_available{false};
    if (placement_req->is_dev_req()) {
      dev_req = std::dynamic_pointer_cast<DeviceRequirement>(placement_req);
      DevID_t dev_global_id = dev_req->device()->get_global_id();
      if (!is_dev_assigned[dev_global_id]) {
        is_member_device_available =
            this->calc_score_devplacement(task, dev_req, mapper, &score,
                parray_list[did]);
        if (is_member_device_available) {
          is_dev_assigned[dev_global_id] = true;
        }
      }
    } else if (placement_req->is_arch_req()) {
      ArchitectureRequirement *arch_req =
          dynamic_cast<ArchitectureRequirement *>(placement_req.get());
      is_member_device_available = this->calc_score_archplacement(
          task, arch_req, mapper, dev_req, &score, parray_list[did],
          &is_dev_assigned);
      if (is_member_device_available) {
        DevID_t dev_global_id = dev_req->device()->get_global_id();
        is_dev_assigned[dev_global_id] = true;
      }
    }
    assert(dev_req != nullptr);
    (*member_device_reqs)[did] = dev_req;
    *average_score += score;
    if (!is_member_device_available) {
      // If any of the device candidates is not available,
      // return false and exclude this option from task mapping.
      return false;
    }
  }
  *average_score /= placement_reqs_vec.size();
  return true;
}
