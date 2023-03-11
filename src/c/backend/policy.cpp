#include "include/policy.hpp"

std::pair<Score_t, Device*> LocalityLoadBalancingMappingPolicy::calc_score_archplacement(
    InnerTask* task, std::shared_ptr<DeviceRequirementBase> base_res_req,
    size_t num_total_mapped_tasks) {
  std::cout
      << "\t[Architecture Requirement in Multi-device Requirement]\n";
  ArchitectureRequirement* arch_res_req =
      dynamic_cast<ArchitectureRequirement *>(base_res_req.get());
  Score_t best_score{0};
  Device* best_device{nullptr};
  uint32_t i = 0;
  for (std::shared_ptr<DeviceRequirement> dev_res_req :
      arch_res_req->GetDeviceRequirementOptions()) {
    auto [score, device] = calc_score_devplacement(task, dev_res_req,
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
  return std::make_pair(best_score, best_device);
}

std::pair<Score_t, Device*>
    LocalityLoadBalancingMappingPolicy::calc_score_devplacement(InnerTask* task,
    std::shared_ptr<DeviceRequirementBase> base_res_req,
    size_t num_total_mapped_tasks /*, std::vector<> num_mapped_tasks_foreach_device */) {
  DeviceRequirement *dev_res_req =
      dynamic_cast<DeviceRequirement *>(base_res_req.get());
  const Device& device = *(dev_res_req->device());
  std::cout << "Locality-aware- and Load-balancing mapping policy\n";

  // TODO(hc): Data locality calculation.
  size_t local_data = 0, nonlocal_data = 0;
  // TODO(hc): PArray loop.

  // size_t dev_load = Get device's number of tasks mapped and running.

  // Check device resource availability.

  // Check device dependencies.
  TaskList& dependencies = task->dependencies;
  // TODO(hc): I am not sure if this is safe since dependencies
  //           can be popped during iteartion?
  //           or maybe we can expose locks to make this
  //           as transaction?
  //           Also, do we consider only alive dependencies?
  //           or do we allow popping dependencies?
  for (size_t i = 0; i < dependencies.size(); ++i) {
    InnerTask* dependency = dependencies.at(i);
    auto dep_placement_reqs = dependency->get_placement_reqs();
    assert(dep_placement_reqs.size() > 0);
    for (std::shared_ptr<DeviceRequirement> req : dep_placement_reqs) {
      std::cout << "Device:" << req->device()->GetName() << "\n";
    }
  }

  Score_t score = 0;
  std::cout << "\t[Device Requirement in Multi-device Requirement]\n"
            << dev_res_req->device()->GetName() << " -> "
            << dev_res_req->res_req().get(MEMORY) << "B, VCU "
            << dev_res_req->res_req().get(VCU) << "\n";
  return std::make_pair(score, dev_res_req->device());
}
