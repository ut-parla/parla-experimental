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

  /// Calculate a score of device placement.
  /// Note that it does not choose a device for a task.
  /// The caller, the task mapper, will choose one of the placement
  /// options.
  ///
  /// @param task Target task for task mapping.
  /// @param base_res_req Resource requirement of the architecture.
  ///                     This is a base type of resource requirement class
  ///                     and, so it will be casted to the proper inherited
  ///                     class within this function.
  /// @return Pair of the chosen device and its score.
  virtual std::pair<Score_t, Device*> calc_score_archplacement(
      InnerTask* task, std::shared_ptr<DeviceRequirementBase> base_res_req) = 0;

  /// Calculate a score of architecture placement.
  /// This function calculate scores for each device of the specified
  /// architecture type, and chooses a device giving the highest score.
  /// Note that it does not choose a device for a task.
  /// The caller, the task mapper, will choose one of the placement
  /// options.
  ///
  /// @param task Target task for task mapping.
  /// @param base_res_req Resource requirement of the architecture.
  ///                     This is a base type of resource requirement class
  ///                     and, so it will be casted to the proper inherited
  ///                     class within this function.
  /// @return Pair of the chosen device and its score.
  virtual std::pair<Score_t, Device*> calc_score_devplacement(
      InnerTask* task, std::shared_ptr<DeviceRequirementBase> base_res_req) = 0;

  const DeviceManager& GetDeviceManagerRef() {
    return *device_manager_;
  }

protected:
  DeviceManager* device_manager_;
};

class LocalityLoadBalancingMappingPolicy : public MappingPolicy {
public:
  using MappingPolicy::MappingPolicy;

  std::pair<Score_t, Device*> calc_score_archplacement(InnerTask* task,
      std::shared_ptr<DeviceRequirementBase> base_res_req) override;

  std::pair<Score_t, Device*> calc_score_devplacement(InnerTask* task,
      std::shared_ptr<DeviceRequirementBase> base_res_req) override;

};

#endif
