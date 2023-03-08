#include "include/policy.hpp"

ScoreTy LocalityLoadBalancingMappingPolicy::CalculateScore(InnerTask* task, DeviceRequirement* dev_res_req) {
  const Device& device = *(dev_res_req->device());
  std::cout << "Locality-aware- and Load-balancing mapping policy\n";
  return 0;
}
