#ifndef PARLA_POLICY_HPP
#define PARLA_POLICY_HPP

#include "device.hpp"
#include "runtime.hpp"

#include <memory>

using ScoreTy = double;

class MappingPolicy {
public:
  virtual ScoreTy CalculateScore(InnerTask*, DeviceRequirement*) = 0;
};

class LocalityLoadBalancingMappingPolicy : public MappingPolicy {
public:
  ScoreTy CalculateScore(InnerTask*, DeviceRequirement*) override;
};

#endif
