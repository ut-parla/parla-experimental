#ifndef PARLA_RESOURCE_REQUIREMENTS_HPP
#define PARLA_RESOURCE_REQUIREMENTS_HPP

#include "device.hpp"

#include <memory>

/// Base classes.

class DeviceRequirementBase {
public:
  virtual bool is_multidev_req() = 0;
  virtual bool is_arch_req() = 0;
  virtual bool is_dev_req() = 0;
};
class SingleDeviceRequirementBase : public DeviceRequirementBase {};

/// Resource contains device types (architectures), specific devices, their
/// memory and virtual computation units.
class ResourceRequirementCollections {
public:
  void
  AppendDeviceRequirementOption(std::shared_ptr<DeviceRequirementBase> dev_req);
  const std::vector<std::shared_ptr<DeviceRequirementBase>> &
  GetDeviceRequirementOptions();

private:
  std::vector<std::shared_ptr<DeviceRequirementBase>> placement_reqs_;
};

class MultiDeviceRequirements : public DeviceRequirementBase {
public:
  void
  AppendDeviceRequirement(std::shared_ptr<SingleDeviceRequirementBase> req);
  bool is_multidev_req() override { return true; }
  bool is_arch_req() override { return false; }
  bool is_dev_req() override { return false; }

  const std::vector<std::shared_ptr<SingleDeviceRequirementBase>>&
      get_placement_requirements();

private:
  std::vector<std::shared_ptr<SingleDeviceRequirementBase>> placement_reqs_;
};

class DeviceRequirement : public SingleDeviceRequirementBase {
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

class ArchitectureRequirement : public SingleDeviceRequirementBase {
public:
  void AppendDeviceRequirementOption(std::shared_ptr<DeviceRequirement> req);
  const std::vector<std::shared_ptr<DeviceRequirement>> &
  GetDeviceRequirementOptions();
  bool is_multidev_req() override { return false; }
  bool is_arch_req() override { return true; }
  bool is_dev_req() override { return false; }

private:
  std::vector<std::shared_ptr<DeviceRequirement>> placement_reqs_;
};

#endif
