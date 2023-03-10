#ifndef PARLA_POLICY_HPP
#define PARLA_POLICY_HPP

#include "device.hpp"
#include "runtime.hpp"

#include <memory>

using Score_t = double;

class MappingPolicy {
public:
  MappingPolicy(DeviceManager* device_manager) :
      device_manager_(device_manager) {}
  virtual Score_t CalculateScore(InnerTask*, DeviceRequirement*) = 0;

  const DeviceManager& GetDeviceManagerRef() {
    return *device_manager_;
  }

protected:
  DeviceManager* device_manager_;
};

class LocalityLoadBalancingMappingPolicy : public MappingPolicy {
public:
  using MappingPolicy::MappingPolicy;

  Score_t CalculateScore(InnerTask*, DeviceRequirement*) override;
};

#endif
