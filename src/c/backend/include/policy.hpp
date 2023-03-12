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
  /// @param num_total_mapped_tasks The total number of tasks mapped to the
  ///                               whole devices.
  /// @return Pair of the chosen device and its score.
  virtual std::pair<Score_t, Device*> calc_score_archplacement(
      InnerTask* task, ArchitectureRequirement *base_res_req,
      size_t num_total_mapped_tasks) = 0;

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
  /// @param num_total_mapped_tasks The total number of tasks mapped to the
  ///                               whole devices.
  /// @return Pair of the chosen device and its score.
  virtual std::pair<Score_t, Device*> calc_score_devplacement(
      InnerTask* task, DeviceRequirement *base_res_req,
      size_t num_total_mapped_tasks) = 0;

  // TODO(hc): comment
  virtual std::pair<Score_t, std::vector<Device*>> calc_score_mdevplacement(
          InnerTask* task, MultiDeviceRequirements *placement_reqs,
          size_t num_total_mapped_tasks) = 0;

  const DeviceManager& GetDeviceManagerRef() {
    return *device_manager_;
  }

protected:
  DeviceManager* device_manager_;
};

class LocalityLoadBalancingMappingPolicy : public MappingPolicy {
public:
  using MappingPolicy::MappingPolicy;

  std::pair<Score_t, Device*> calc_score_archplacement(
      InnerTask *task, ArchitectureRequirement *base_res_req,
      size_t num_total_mapped_tasks) override;

  std::pair<Score_t, Device*> calc_score_devplacement(
      InnerTask *task, DeviceRequirement *base_res_req,
      size_t num_total_mapped_tasks) override;

  std::pair<Score_t, std::vector<Device*>> calc_score_mdevplacement(
          InnerTask* task, MultiDeviceRequirements *placement_reqs,
          size_t num_total_mapped_tasks) override;
};

#endif
