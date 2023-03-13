#include "include/resource_requirements.hpp"

void ResourceRequirementCollections::AppendDeviceRequirementOption(
    std::shared_ptr<DeviceRequirementBase> req) {
  placement_reqs_.emplace_back(std::shared_ptr<DeviceRequirementBase>(req));
}

const std::vector<std::shared_ptr<DeviceRequirementBase>> &
ResourceRequirementCollections::GetDeviceRequirementOptions() {
  return placement_reqs_;
}

void MultiDeviceRequirements::AppendDeviceRequirement(
    std::shared_ptr<SingleDeviceRequirementBase> req) {
  placement_reqs_.emplace_back(
      std::shared_ptr<SingleDeviceRequirementBase>(req));
}

const std::vector<std::shared_ptr<SingleDeviceRequirementBase>> &
MultiDeviceRequirements::get_placement_requirements_ref() {
  return placement_reqs_;
}

void ArchitectureRequirement::AppendDeviceRequirementOption(
    std::shared_ptr<DeviceRequirement> req) {
  placement_reqs_.emplace_back(std::shared_ptr<DeviceRequirement>(req));
}

const std::vector<std::shared_ptr<DeviceRequirement>> &
ArchitectureRequirement::GetDeviceRequirementOptions() {
  return placement_reqs_;
}
