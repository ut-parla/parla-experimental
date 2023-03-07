#include "include/resource_requirements.hpp"

void ResourceRequirementCollections::AppendDeviceRequirementOption(
    std::shared_ptr<DeviceRequirementBase> req) {
  dev_reqs_.emplace_back(std::shared_ptr<DeviceRequirementBase>(req));
}

const std::vector<std::shared_ptr<DeviceRequirementBase>>&
    ResourceRequirementCollections::GetDeviceRequirementOptions() {
  return dev_reqs_;
}

void MultiDeviceRequirements::AppendDeviceRequirement(
    std::shared_ptr<SingleDeviceRequirementBase> req) {
  dev_reqs_.emplace_back(std::shared_ptr<SingleDeviceRequirementBase>(req));
}

const std::vector<std::shared_ptr<SingleDeviceRequirementBase>>&
    MultiDeviceRequirements::GetDeviceRequirements() {
  return dev_reqs_;
}

void ArchitectureRequirement::AppendDeviceRequirementOption(
    std::shared_ptr<DeviceRequirement> req) {
  dev_reqs_.emplace_back(std::shared_ptr<DeviceRequirement>(req));
}


const std::vector<std::shared_ptr<DeviceRequirement>>&
ArchitectureRequirement::GetDeviceRequirementOptions() {
  return dev_reqs_;
}
