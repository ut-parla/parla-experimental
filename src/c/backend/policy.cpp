#include "include/policy.hpp"

std::pair<Score_t, Device*> LocalityLoadBalancingMappingPolicy::calc_score_archplacement(
    InnerTask* task, std::shared_ptr<DeviceRequirementBase> base_res_req) {
  std::cout
      << "\t[Architecture Requirement in Multi-device Requirement]\n";
  ArchitectureRequirement* arch_res_req =
      dynamic_cast<ArchitectureRequirement *>(base_res_req.get());
  Score_t best_score{0};
  Device* best_device{nullptr};
  uint32_t i = 0;
  for (std::shared_ptr<DeviceRequirement> dev_res_req :
      arch_res_req->GetDeviceRequirementOptions()) {
    auto [score, device] = calc_score_devplacement(task, dev_res_req);
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
  return std::make_pair(best_score, best_device);
}

std::pair<Score_t, Device*>
    LocalityLoadBalancingMappingPolicy::calc_score_devplacement(InnerTask* task,
    std::shared_ptr<DeviceRequirementBase> base_res_req) {
  DeviceRequirement *dev_res_req =
      dynamic_cast<DeviceRequirement *>(base_res_req.get());
  const Device& device = *(dev_res_req->device());
  std::cout << "Locality-aware- and Load-balancing mapping policy\n";

  size_t num_total_mapped_tasks = GetDeviceManagerRef().TotalNumMappedTasks();

  // TODO(hc): Data locality calculation.
  size_t local_data = 0, nonlocal_data = 0;
  // TODO(hc): PArray loop.

  // size_t dev_load = Get device's number of tasks mapped and running.

  // Check device resource availability.

  // Check device dependencies.

  Score_t score = 0;
  std::cout << "\t[Device Requirement in Multi-device Requirement]\n"
            << dev_res_req->device()->GetName() << " -> "
            << dev_res_req->res_req().get(MEMORY) << "B, VCU "
            << dev_res_req->res_req().get(VCU) << "\n";
  return std::make_pair(score, dev_res_req->device());
}
