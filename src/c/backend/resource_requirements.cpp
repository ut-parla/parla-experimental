#include "include/resource_requirements.hpp"

void ResourceRequirementCollections::AppendDeviceRequirementOption(
    DeviceRequirementBase* req) {
  dev_reqs_.emplace_back(req);
}

const std::vector<DeviceRequirementBase*>&
    ResourceRequirementCollections::GetDeviceRequirementOptions() {
  return dev_reqs_;
}

void MultiDeviceRequirements::AppendDeviceRequirement(
    SingleDeviceRequirementBase* req) {
  dev_reqs_.emplace_back(req);
}

const std::vector<SingleDeviceRequirementBase*>&
    MultiDeviceRequirements::GetDeviceRequirements() {
  return dev_reqs_;
}

void ArchitectureRequirement::AppendDeviceRequirementOption(
    DeviceRequirement* req) {
  dev_reqs_.emplace_back(req);
}


const std::vector<DeviceRequirement*>&
ArchitectureRequirement::GetDeviceRequirementOptions() {
  return dev_reqs_;
}
