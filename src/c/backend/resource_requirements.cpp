#include "include/resource_requirements.hpp"

void ResourceRequirementCollections::AppendDeviceRequirementOption(
    DeviceRequirementBase* req) {
  dev_reqs_.emplace_back(req);
}

void MultiDeviceRequirements::AppendDeviceRequirement(
    SingleDeviceRequirementBase* req) {
  dev_reqs_.emplace_back(req);
}

void ArchitectureRequirement::AppendDeviceRequirementOption(
    DeviceRequirement* req) {
  dev_reqs_.emplace_back(req);
}
