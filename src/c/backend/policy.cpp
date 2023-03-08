#include "include/policy.hpp"

ScoreTy LocalityLoadBalancingMappingPolicy::CalculateScore(InnerTask* task, DeviceRequirement* dev_res_req) {
  const Device& device = *(dev_res_req->device());
  std::cout << "Locality-aware- and Load-balancing mapping policy\n";

  size_t num_total_mapped_tasks = GetDeviceManagerRef().TotalNumMappedTasks();

  // TODO(hc): Data locality calculation.
  size_t local_data = 0, nonlocal_data = 0;
  // TODO(hc): PArray loop.

  // size_t dev_load = Get device's number of tasks mapped and running.

  // Check device resource availability.

  // Check device dependencies.

  std::cout << "\t>>" << num_total_mapped_tasks << "\n";
  return 0;
}
