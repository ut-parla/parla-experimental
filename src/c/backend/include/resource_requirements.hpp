#ifndef PARLA_RESOURCE_REQUIREMENTS_HPP
#define PARLA_RESOURCE_REQUIREMENTS_HPP

#include "device.hpp"

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
  void AppendDeviceRequirementOption(DeviceRequirementBase* dev_req);
  const std::vector<DeviceRequirementBase*>& GetDeviceRequirementOptions();

private:
  std::vector<DeviceRequirementBase*> dev_reqs_;
};

class MultiDeviceRequirements : public DeviceRequirementBase {
public:
  void AppendDeviceRequirement(SingleDeviceRequirementBase* req);
  const std::vector<SingleDeviceRequirementBase*>& GetDeviceRequirements();
  bool is_multidev_req() override { return true; }
  bool is_arch_req() override { return false; }
  bool is_dev_req() override { return false; }

private:
  std::vector<SingleDeviceRequirementBase*> dev_reqs_;
};

class DeviceRequirement : public SingleDeviceRequirementBase {
public:
  DeviceRequirement(Device* dev, DeviceResources res_reqs)
      : dev_(dev), res_reqs_(res_reqs) {}

  bool is_multidev_req() override { return false; }
  bool is_arch_req() override { return false; }
  bool is_dev_req() override { return true; }

  const Device& device() { return (*dev_); }

  const DeviceResources& res_req() { return res_reqs_; }

private:
  Device* dev_;
  DeviceResources res_reqs_;
};

class ArchitectureRequirement : public SingleDeviceRequirementBase {
public:
  void AppendDeviceRequirementOption(DeviceRequirement* req);
  const std::vector<DeviceRequirement*>& GetDeviceRequirementOptions();
  bool is_multidev_req() override { return false; }
  bool is_arch_req() override { return true; }
  bool is_dev_req() override { return false; }

private:
  std::vector<DeviceRequirement*> dev_reqs_;
};

#endif
