#include "include/resource_requirements.hpp"

void MultiDeviceRequirements::AppendDeviceRequirement(
    SingleDeviceRequirementBase* req) {
  dev_reqs_.emplace_back(req);
}

void ArchitectureRequirement::AppendDeviceRequirementOption(
    DeviceRequirement* req) {
  dev_reqs_.emplace_back(req);
}
