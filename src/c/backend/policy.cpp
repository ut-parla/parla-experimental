#include "include/policy.hpp"

std::pair<Score_t, Device*>
    LocalityLoadBalancingMappingPolicy::calc_score_archplacement(
        InnerTask* task, ArchitectureRequirement* arch_placement_req,
        size_t num_total_mapped_tasks) {
  std::cout
      << "\t[Architecture Requirement in Multi-device Requirement]\n";
  Score_t best_score{0};
  Device* best_device{nullptr};
  uint32_t i = 0;
  for (std::shared_ptr<DeviceRequirement> dev_res_req :
      arch_placement_req->GetDeviceRequirementOptions()) {
    auto [score, device] = calc_score_devplacement(task, dev_res_req.get(),
        num_total_mapped_tasks);
    if (best_score < score) {
      best_score = score;
      best_device = device;
    }
    std::cout << "\t\t[" << i << "]"
              << device->GetName() << " -> "
              << dev_res_req->res_req().get(MEMORY) << "B, VCU "
              << dev_res_req->res_req().get(VCU) << "\n";
    ++i;
  }
  assert(best_device != nullptr);
  return std::make_pair(best_score, best_device);
}

std::pair<Score_t, Device*>
    LocalityLoadBalancingMappingPolicy::calc_score_devplacement(
        InnerTask* task, DeviceRequirement* placement_req,
        size_t num_total_mapped_tasks
        /*, std::vector<> num_mapped_tasks_foreach_device */) {
  const Device& device = *(placement_req->device());
  std::cout << "Locality-aware- and Load-balancing mapping policy\n";

  // TODO(hc): Data locality calculation.
  size_t local_data = 0, nonlocal_data = 0;
  // TODO(hc): PArray loop.

  // size_t dev_load = Get device's number of tasks mapped and running.

  // Check device resource availability.

  // Check device dependencies.
  TaskList& dependencies = task->dependencies;
  for (size_t i = 0; i < dependencies.size_unsafe(); ++i) {
    InnerTask* dependency = dependencies.at_unsafe(i);
    auto dep_placement_reqs = dependency->get_placement_reqs();
    assert(dep_placement_reqs.size() > 0);
    for (std::shared_ptr<DeviceRequirement> req : dep_placement_reqs) {
      std::cout << "Device:" << req->device()->GetName() << "\n";
    }
  }

  Score_t score = 0;
  std::cout << "\t[Device Requirement in Multi-device Requirement]\n"
            << placement_req->device()->GetName() << " -> "
            << placement_req->res_req().get(MEMORY) << "B, VCU "
            << placement_req->res_req().get(VCU) << "\n";
  return std::make_pair(score, placement_req->device());
}

std::pair<Score_t, std::vector<Device*>>
    LocalityLoadBalancingMappingPolicy::calc_score_mdevplacement(
        InnerTask* task, MultiDeviceRequirements *placement_reqs,
        size_t num_total_mapped_tasks) {
  Score_t average_score{0};
  std::vector<std::shared_ptr<SingleDeviceRequirementBase>> placement_reqs_vec =
         placement_reqs->get_placement_requirements();
  std::vector<Device*> member_devices(placement_reqs_vec.size());
  for (DevID_t did = 0; did < placement_reqs_vec.size(); ++did) {
    std::shared_ptr<SingleDeviceRequirementBase> placement_req =
        placement_reqs_vec[did];
    Device* dev{nullptr};
    Score_t score{0};
    if (placement_req->is_dev_req()) {
      DeviceRequirement *dev_req =
          dynamic_cast<DeviceRequirement *>(placement_req.get());
      std::tie(score, dev) = this->calc_score_devplacement(task, dev_req,
          num_total_mapped_tasks);
    } else if (placement_req->is_arch_req()) {
      ArchitectureRequirement* arch_req =
          dynamic_cast<ArchitectureRequirement *>(placement_req.get());
      std::tie(score, dev) =
          this->calc_score_archplacement(task, arch_req,
              num_total_mapped_tasks);
    }
    assert(dev != nullptr);
    member_devices[did] = dev;
    average_score += score;
  }
  average_score /= placement_reqs_vec.size();
  return std::make_pair(average_score, member_devices);
}

