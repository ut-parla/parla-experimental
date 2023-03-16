#include "include/policy.hpp"

bool LocalityLoadBalancingMappingPolicy::calc_score_devplacement(
    InnerTask *task,
    const std::shared_ptr<DeviceRequirement> &dev_placement_req,
    size_t num_total_mapped_tasks, Score_t *score
    /*, std::vector<> num_mapped_tasks_foreach_device */) {
  const Device &device = *(dev_placement_req->device());
  std::cout << "[Locality-aware- and Load-balancing mapping policy]\n";
  std::cout << " ** Device Score Calculation\n";

  // TODO(hc): Data locality calculation.
  size_t local_data = 0, nonlocal_data = 0;
  // TODO(hc): PArray loop.

  // size_t dev_load = Get device's number of tasks mapped and running.

  // Check device resource availability.
  if (!device.check_resource_availability()) {
    return false;
  }

  // Check device dependencies.
  TaskList &dependencies = task->dependencies;
  for (size_t i = 0; i < dependencies.size_unsafe(); ++i) {
    InnerTask *dependency = dependencies.at_unsafe(i);
    auto dep_placement_reqs = dependency->get_placement_reqs();
    assert(dep_placement_reqs.size() > 0);
  }

  *score = 0;
  std::cout << "\t[Device Requirement in device Requirement]\n"
            << "\t\t" << dev_placement_req->device()->get_name() << " -> "
            << dev_placement_req->res_req().get(Resource::Memory) << "B, VCU "
            << dev_placement_req->res_req().get(Resource::VCU) << "\n";
  return true;
}

bool LocalityLoadBalancingMappingPolicy::calc_score_archplacement(
    InnerTask *task, ArchitectureRequirement *arch_placement_req,
    size_t num_total_mapped_tasks,
    std::shared_ptr<DeviceRequirement> &chosen_dev_req,
    Score_t *chosen_dev_score) {
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
    bool is_dev_available = this->calc_score_devplacement(
        task, dev_req, num_total_mapped_tasks, &score);
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
    size_t num_total_mapped_tasks,
    std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
    Score_t *average_score) {
  *average_score = 0;
  const std::vector<std::shared_ptr<SinglePlacementRequirementBase>>
      &placement_reqs_vec = mdev_placement_req->get_placement_reqs_ref();
  member_device_reqs->resize(placement_reqs_vec.size());
  // Iterate requirements of the devices specified in multi-device placement.
  // All of the member devices should be available.
  for (DevID_t did = 0; did < placement_reqs_vec.size(); ++did) {
    std::shared_ptr<SinglePlacementRequirementBase> placement_req =
        placement_reqs_vec[did];
    std::shared_ptr<DeviceRequirement> dev_req{nullptr};
    Score_t score{0};
    bool is_member_device_available{true};
    if (placement_req->is_dev_req()) {
      dev_req = std::dynamic_pointer_cast<DeviceRequirement>(placement_req);
      is_member_device_available = this->calc_score_devplacement(
          task, dev_req, num_total_mapped_tasks, &score);
    } else if (placement_req->is_arch_req()) {
      ArchitectureRequirement *arch_req =
          dynamic_cast<ArchitectureRequirement *>(placement_req.get());
      is_member_device_available = this->calc_score_archplacement(
          task, arch_req, num_total_mapped_tasks, dev_req, &score);
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
