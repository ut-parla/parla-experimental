#ifndef PARLA_POLICY_HPP
#define PARLA_POLICY_HPP

#include "device.hpp"
#include "parray_tracker.hpp"
#include "runtime.hpp"

#include <memory>

using Score_t = double;

class Mapper;

class MappingPolicy {
public:
  MappingPolicy(DeviceManager *device_manager, PArrayTracker *parray_tracker)
      : device_manager_(device_manager), parray_tracker_(parray_tracker) {}

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
  /// @param mapper Mapper instance to get mapping information.
  /// @param chosen_dev_score A pointer of a score of the device.
  /// @return True if a device is available
  virtual bool calc_score_devplacement(
      InnerTask *task,
      const std::shared_ptr<DeviceRequirement> &dev_placement_req,
      const Mapper &mapper, Score_t *score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
              &parray_list) = 0;

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
  /// @param mapper Mapper instance to get mapping information.
  /// @param chosen_dev_req A pointer of a chosen device and its resource
  ///                       requirement. This is a reference type since
  ///                       this function chooses a device and updates its
  ///                       pointer to the device requirement.
  /// @param chosen_dev_score A pointer of a score of the chosen device.
  /// @return True if any device in the architecture is available
  virtual bool calc_score_archplacement(
      InnerTask *task, ArchitectureRequirement *arch_placement_req,
      const Mapper &mapper, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
      Score_t *chosen_dev_score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
              &parray_list) = 0;

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
  /// @param mapper Mapper instance to get mapping information.
  /// @param member_device_reqs A vector of the resource requirement of the
  ///                           member device.
  /// @param chosen_dev_score A pointer of a score of the multiple devices.
  /// @return True if all devices in the multi-device placement are available.
  virtual bool calc_score_mdevplacement(
      InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
      const Mapper &mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
      Score_t *average_score,
      const std::vector<
          std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
              &parray_list) = 0;

protected:
  DeviceManager *device_manager_;
  PArrayTracker *parray_tracker_;
};

class LocalityLoadBalancingMappingPolicy : public MappingPolicy {
public:
  using MappingPolicy::MappingPolicy;

  bool calc_score_devplacement(
      InnerTask *task,
      const std::shared_ptr<DeviceRequirement> &dev_placement_req,
      const Mapper &mapper, Score_t *score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
              &parray_list) override;

  bool calc_score_archplacement(
      InnerTask *task, ArchitectureRequirement *arch_placement_req,
      const Mapper &mapper, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
      Score_t *chosen_dev_score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
              &parray_list) override;

  bool calc_score_mdevplacement(
      InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
      const Mapper &mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
      Score_t *average_score,
      const std::vector<
          std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
              &parray_list) override;
};

#endif
