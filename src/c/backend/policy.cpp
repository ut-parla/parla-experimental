#include "include/policy.hpp"

void LocalityLoadBalancingMappingPolicy::calc_score_devplacement(
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
            << dev_placement_req->res_req().get(MEMORY) << "B, VCU "
            << dev_placement_req->res_req().get(VCU) << "\n";
}

void LocalityLoadBalancingMappingPolicy::calc_score_archplacement(
    InnerTask *task, ArchitectureRequirement *arch_placement_req,
    size_t num_total_mapped_tasks,
    std::shared_ptr<DeviceRequirement> &chosen_dev_req,
    Score_t *chosen_dev_score) {
  std::cout << "** Architecture Requirement in Multi-device Requirement\n";
  Score_t best_score{0};
  std::shared_ptr<DeviceRequirement> best_device_req{nullptr};
  uint32_t i = 0;
  // For now the architecture placement has one resource requirement
  // regardless of the devices of the architecture. In the future,
  // we will allow a separate placement for each device.
  for (std::shared_ptr<DeviceRequirement> dev_req :
       arch_placement_req->GetDeviceRequirementOptions()) {
    Score_t score{0};
    this->calc_score_devplacement(task, dev_req, num_total_mapped_tasks,
                                  &score);
    if (best_score <= score) {
      best_score = score;
      best_device_req = dev_req;
    }
    ++i;
  }
  assert(best_device_req != nullptr);
  chosen_dev_req = best_device_req;
  *chosen_dev_score = best_score;
}

void LocalityLoadBalancingMappingPolicy::calc_score_mdevplacement(
    InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
    size_t num_total_mapped_tasks,
    std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
    Score_t *average_score) {
  *average_score = 0;
  const std::vector<std::shared_ptr<SinglePlacementRequirementBase>> &
      placement_reqs_vec = mdev_placement_req->get_placement_reqs_ref();
  member_device_reqs->resize(placement_reqs_vec.size());
  for (DevID_t did = 0; did < placement_reqs_vec.size(); ++did) {
    std::shared_ptr<SinglePlacementRequirementBase> placement_req =
        placement_reqs_vec[did];
    std::shared_ptr<DeviceRequirement> dev_req{nullptr};
    Score_t score{0};
    if (placement_req->is_dev_req()) {
      dev_req = std::dynamic_pointer_cast<DeviceRequirement>(placement_req);
      this->calc_score_devplacement(task, dev_req, num_total_mapped_tasks,
                                    &score);
    } else if (placement_req->is_arch_req()) {
      ArchitectureRequirement *arch_req =
          dynamic_cast<ArchitectureRequirement *>(placement_req.get());
      this->calc_score_archplacement(task, arch_req, num_total_mapped_tasks,
                                     dev_req, &score);
    }
    assert(dev_req != nullptr);
    (*member_device_reqs)[did] = dev_req;
    *average_score += score;
  }
  *average_score /= placement_reqs_vec.size();
}
