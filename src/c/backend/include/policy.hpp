#ifndef PARLA_POLICY_HPP
#define PARLA_POLICY_HPP

#include "device.hpp"
#include "runtime.hpp"

#include <memory>

class MappingPolicy {
public:
  virtual void MapTask(InnerTask*, const Device&) = 0;
};

class LocalityLoadBalancingMappingPolicy : public MappingPolicy {
public:
  void MapTask(InnerTask*, const Device&) override;
};

#endif
