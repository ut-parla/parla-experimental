#ifndef PARLA_RESOURCE_REQUIREMENTS_HPP
#define PARLA_RESOURCE_REQUIREMENTS_HPP

#include "device.hpp"

/// Base classes.

class DeviceRequirementBase {};
class SingleDeviceRequirementBase : DeviceRequirementBase {};

using MemorySzTy = uint64_t;
using VCUTy = uint32_t;

/// Resource contains device types (architectures), specific devices, their
/// memory and virtual computation units.
class ResourceRequirementCollections {
public:
private:
  std::vector<DeviceRequirementBase*> dev_reqs_;
};

class MultiDeviceRequirements : DeviceRequirementBase {
  void AppendDeviceRequirement(SingleDeviceRequirementBase* req);
private:
  std::vector<SingleDeviceRequirementBase*> dev_reqs_;
};

class DeviceRequirement : public SingleDeviceRequirementBase {
public:
  DeviceRequirement() = delete;
  DeviceRequirement(Device* dev, DeviceResources res_reqs) :
      dev_(dev), res_reqs_(res_reqs) {}
private:
  Device* dev_;
  DeviceResources res_reqs_;
};

class ArchitectureRequirement : public SingleDeviceRequirementBase {
  void AppendDeviceRequirementOption(DeviceRequirement* req);
private:
  std::vector<DeviceRequirement*> dev_reqs_;
};

#endif
