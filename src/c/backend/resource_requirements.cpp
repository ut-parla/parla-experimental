#include "include/resource_requirements.hpp"

void PlacementRequirementCollections::append_placement_req_opt(
    std::shared_ptr<PlacementRequirementBase> req) {
  placement_reqs_.emplace_back(std::shared_ptr<PlacementRequirementBase>(req));
}

const std::vector<std::shared_ptr<PlacementRequirementBase>> &
PlacementRequirementCollections::get_placement_req_opts_ref() {
  return placement_reqs_;
}

void MultiDeviceRequirements::append_placement_req(
    std::shared_ptr<SinglePlacementRequirementBase> req) {
  placement_reqs_.emplace_back(
      std::shared_ptr<SinglePlacementRequirementBase>(req));
}

const std::vector<std::shared_ptr<SinglePlacementRequirementBase>> &
MultiDeviceRequirements::get_placement_reqs_ref() {
  return placement_reqs_;
}

void ArchitectureRequirement::append_placement_req_opt(
    std::shared_ptr<DeviceRequirement> req) {
  placement_reqs_.emplace_back(std::shared_ptr<DeviceRequirement>(req));
}

const std::vector<std::shared_ptr<DeviceRequirement>> &
ArchitectureRequirement::GetDeviceRequirementOptions() {
  return placement_reqs_;
}
