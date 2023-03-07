#include "include/policy.hpp"

void LocalityLoadBalancingMappingPolicy::MapTask(InnerTask*, const Device&) {
  std::cout << "Locality-aware- and Load-balancing mapping policy\n";
}
