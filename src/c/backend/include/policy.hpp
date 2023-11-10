/*! @file policy.hpp
 *  @brief Interface for mapping policies.
 */

#ifndef PARLA_POLICY_HPP
#define PARLA_POLICY_HPP

#include "device.hpp"
#include "parray_tracker.hpp"
#include "runtime.hpp"

#include <memory>

/// The type of a suitability score of a device
using Score_t = double;

class Mapper;

/*!
 * @brief Interface for mapping (task to device assignment) policies.
 */
class MappingPolicy {
public:
  MappingPolicy(DeviceManager *device_manager, PArrayTracker *parray_tracker)
      : device_manager_(device_manager), parray_tracker_(parray_tracker) {}

  /// @brief Calculate a score of the device placement requirement.
  /// @details This function calculates a score of a device based on the current
  /// states of the device (e.g., available memory and the number of vcus).
  /// It returns a device giving the best score and its score.
  /// Note that it does not choose a device for a task, but
  /// the caller, so the task mapper, will choose one of the placement
  /// options.
  ///
  /// @param task Target task for task mapping.
  /// @param dev_placement_req Resource requirement of the device.
  /// @param mapper Mapper instance to get mapping information.
  /// @param score A pointer of a score of the device.
  /// @param parray_list A list of PArray instances used by the target task
  /// @return True if a device is available
  virtual bool calc_score_devplacement(
      InnerTask *task,
      const std::shared_ptr<DeviceRequirement> &dev_placement_req,
      const Mapper &mapper, Score_t *score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
          &parray_list) = 0;

  /// @brief Calculate a score of the architecture placement requirement.
  /// @details This function first iterates devices of the architecture, and
  /// calculates a score for each device based on the current states of the
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
  /// @param parray_list A list of PArray instances used by the target task
  /// @param is_dev_assigned Multi-device task is not allowed to be assigned
  ///                        to duplicated devices. This vector marks
  ///                        assigned devices and avoid that case.
  /// @return True if any device in the architecture is available
  virtual bool calc_score_archplacement(
      InnerTask *task, ArchitectureRequirement *arch_placement_req,
      const Mapper &mapper, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
      Score_t *chosen_dev_score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
          &parray_list,
      std::vector<bool> *is_dev_assigned = nullptr) = 0;

  /// @brief Calculate a score of the multi-device placement that users passed.
  /// @details The placement requirement could contain multiple device or/and
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
  /// @param parray_list A list of PArray instances used by the target task
  /// @return True if all devices in the multi-device placement are available.
  virtual bool calc_score_mdevplacement(
      InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
      const Mapper &mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
      Score_t *average_score,
      const std::vector<
          std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list) = 0;

  virtual void run_task_mapping(
      InnerTask *task, const Mapper &mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *chosen_devices,
      const std::vector<
          std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list,
      std::vector<std::shared_ptr<PlacementRequirementBase>>
          *placement_req_options_vec) = 0;

protected:
  DeviceManager *device_manager_;
  PArrayTracker *parray_tracker_;
  int rrcount = 0;
};

/*!
 *
 */
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
          &parray_list,
      std::vector<bool> *is_dev_assigned = nullptr) override;

  bool calc_score_mdevplacement(
      InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
      const Mapper &mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
      Score_t *average_score,
      const std::vector<
          std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list) override;

  void run_task_mapping(
      InnerTask *task, const Mapper &mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *chosen_devices,
      const std::vector<
          std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list,
      std::vector<std::shared_ptr<PlacementRequirementBase>>
          *placement_req_options_vec) override;
};

#endif
