#ifndef PARLA_RESOURCE_REQUIREMENTS_HPP
#define PARLA_RESOURCE_REQUIREMENTS_HPP

/// Base classes.

class ResourceRequirementBase {};
class SingleDeviceRequirementBase : ResourceRequirementBase {};

/// Resource contains device types (architectures), specific devices, their
/// memory and virtual computation units.
class ResourceRequirementCollections {
public:
private:
  std::vector<ResourceRequirementBase*> reqs_;
};

class MultiDeviceRequirements : ResourceRequirementBase{
private:
  std::vector<SingleDeviceRequirementBase*> reqs_;
};

class DeviceRequirement : public SingleDeviceRequirementBase {

};

class ArchitectureRequirement : public SingleDeviceRequirementBase {
private:
  std::vector<DeviceRequirement*> optional_reqs_;
};

#endif
