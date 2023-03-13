#ifndef PARLA_RESOURCE_REQUIREMENTS_HPP
#define PARLA_RESOURCE_REQUIREMENTS_HPP

#include "device.hpp"

#include <memory>

/// Base classes.

class PlacementRequirementBase {
public:
  virtual bool is_multidev_req() = 0;
  virtual bool is_arch_req() = 0;
  virtual bool is_dev_req() = 0;
};
class SinglePlacementRequirementBase : public PlacementRequirementBase {};

/// Resource contains device types (architectures), specific devices, their
/// memory and virtual computation units.
class PlacementRequirementCollections {
public:
  void append_placement_req_opt(std::shared_ptr<PlacementRequirementBase> dev_req);
  const std::vector<std::shared_ptr<PlacementRequirementBase>> &
  get_placement_req_opts_ref();

private:
  std::vector<std::shared_ptr<PlacementRequirementBase>> placement_reqs_;
};

class MultiDeviceRequirements : public PlacementRequirementBase {
public:
  void
  append_placement_req(std::shared_ptr<SinglePlacementRequirementBase> req);
  bool is_multidev_req() override { return true; }
  bool is_arch_req() override { return false; }
  bool is_dev_req() override { return false; }

  const std::vector<std::shared_ptr<SinglePlacementRequirementBase>> &
  get_placement_reqs_ref();

private:
  std::vector<std::shared_ptr<SinglePlacementRequirementBase>> placement_reqs_;
};

class DeviceRequirement : public SinglePlacementRequirementBase {
public:
  DeviceRequirement(Device *dev, ResourcePool_t res_reqs)
      : dev_(dev), res_reqs_(res_reqs) {}

  bool is_multidev_req() override { return false; }
  bool is_arch_req() override { return false; }
  bool is_dev_req() override { return true; }

  Device *device() { return dev_; }

  const ResourcePool_t &res_req() { return res_reqs_; }

private:
  Device *dev_;
  ResourcePool_t res_reqs_;
};

class ArchitectureRequirement : public SinglePlacementRequirementBase {
public:
  void append_placement_req_opt(std::shared_ptr<DeviceRequirement> req);
  const std::vector<std::shared_ptr<DeviceRequirement>> &
  GetDeviceRequirementOptions();
  bool is_multidev_req() override { return false; }
  bool is_arch_req() override { return true; }
  bool is_dev_req() override { return false; }

private:
  std::vector<std::shared_ptr<DeviceRequirement>> placement_reqs_;
};

#endif
