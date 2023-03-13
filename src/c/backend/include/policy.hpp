#ifndef PARLA_POLICY_HPP
#define PARLA_POLICY_HPP

#include "device.hpp"
#include "runtime.hpp"

#include <memory>

using Score_t = double;

class MappingPolicy {
public:
  MappingPolicy(DeviceManager *device_manager) :
      device_manager_(device_manager) {}

  /// Calculate a score of the device placement requirement.
  /// This function calculates a score of a device based on the current
  /// states of the device (e.g., available memory and the number of vcus).
  /// It returns a device giving the best score and its score.
  /// Note that it does not choose a device for a task, but
  /// the caller, so the task mapper, will choose one of the placement
  /// options.
  ///
  /// @param task Target task for task mapping.
  /// @param dev_placement_req Resource requirement of the device.
  /// @param num_total_mapped_tasks The total number of tasks mapped to the
  ///                               whole devices. TODO(hc): will be packed these
  ///                               kind of factor information to a new wrapper.
  /// @param chosen_dev_score A pointer of a score of the device.
  virtual void calc_score_devplacement(InnerTask* task,
      const std::shared_ptr<DeviceRequirement>& dev_placement_req,
      size_t num_total_mapped_tasks, Score_t *score) = 0;

  /// Calculate a score of the architecture placement requirement.
  /// This function first iterates devices of the architecture, and calculates
  /// a score for each device based on the current states of the
  /// device (e.g., available memory and the number of vcus).
  /// It returns a device giving the best score and its score.
  /// Note that it does not choose a device for a task, but
  /// the caller, so the task mapper, will choose one of the placement
  /// options.
  ///
  /// @param task Target task for task mapping.
  /// @param arch_placement_req Resource requirement of the architecture.
  /// @param num_total_mapped_tasks The total number of tasks mapped to the
  ///                               whole devices. TODO(hc): will pack these
  ///                               kind of factor information to a new wrapper.
  /// @param chosen_dev_req A pointer of a chosen device and its resource
  ///                       requirement. This is a reference type since
  ///                       this function chooses a device and updates its
  ///                       pointer to the device requirement.
  /// @param chosen_dev_score A pointer of a score of the chosen device.
  virtual void calc_score_archplacement(InnerTask *task,
          ArchitectureRequirement *arch_placement_req,
          size_t num_total_mapped_tasks,
          std::shared_ptr<DeviceRequirement> &chosen_dev_req,
          Score_t *chosen_dev_score) = 0;

  /// Calculate a score of the multi-device placement that users passed. 
  /// The placement requirement could contain multiple device or/and
  /// architecture requirements.
  /// This function calculates a score for each placement requirement by
  /// recursively calling a device or an architecture score calculation
  /// function and averages those scores.
  /// This average is used as a score of the multi-device placement
  /// and the caller, so the task mapper, will choose one of the placement
  /// options that give the best score.
  ///
  /// @param task Target task for task mapping.
  /// @param mdev_placement_req Resource requirement of the multiple devices.
  /// @param num_total_mapped_tasks The total number of tasks mapped to the
  ///                               whole devices. TODO(hc): will pack these
  ///                               kind of factor information to a new wrapper.
  /// @param member_device_reqs A vector of the resource requirement of the
  ///                           member device.
  /// @param chosen_dev_score A pointer of a score of the multiple devices.
  virtual void calc_score_mdevplacement(
      InnerTask* task, MultiDeviceRequirements *mdev_placement_req,
      size_t num_total_mapped_tasks,
      std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
      Score_t *average_score) = 0;

protected:
  DeviceManager* device_manager_;
};

class LocalityLoadBalancingMappingPolicy : public MappingPolicy {
public:
  using MappingPolicy::MappingPolicy;

  void calc_score_devplacement(InnerTask* task,
      const std::shared_ptr<DeviceRequirement>& dev_placement_req,
      size_t num_total_mapped_tasks, Score_t *score) override;

  void calc_score_archplacement(InnerTask* task,
      ArchitectureRequirement *arch_placement_req,
      size_t num_total_mapped_tasks,
      std::shared_ptr<DeviceRequirement> &chosen_dev_req,
      Score_t *chosen_dev_score) override;

  void calc_score_mdevplacement(
      InnerTask* task, MultiDeviceRequirements *mdev_placement_req,
      size_t num_total_mapped_tasks,
      std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
      Score_t *average_score) override;
};

#endif
