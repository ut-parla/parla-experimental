#include "include/rl_task_mapper.hpp"

RLTaskMappingPolicy::RLTaskMappingPolicy(
    DeviceManager *device_manager, PArrayTracker *parray_tracker,
    Mapper *mapper)
    : MappingPolicy(device_manager, parray_tracker) {
  size_t num_devices = device_manager->get_num_devices();
  this->rl_agent_ = new RLAgent(num_devices, num_devices, num_devices);
  this->rl_env_ = new RLEnvironment(this->device_manager_, mapper);
}

bool RLTaskMappingPolicy::calc_score_devplacement(
    InnerTask *task,
    const std::shared_ptr<DeviceRequirement> &dev_placement_req,
    const Mapper &mapper, Score_t *score,
    const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
              &parray_list) {
#if 0
  const ParlaDevice &device = *(dev_placement_req->device());
  DevID_t global_dev_id = device.get_global_id();

  // Check device resource availability.
  if (!device.check_resource_availability(dev_placement_req.get())) {
    // std::cout << "Device resource failure!" << std::endl;
    return false;
  }

  auto [chosen_device, chosen_device_score] = this->rl_agent_->select_device(
      this->rl_current_state_,
      this->device_manager_->template get_devices<ParlaDeviceType::All>());
  *score = chosen_device_score;
  /*
  torch::Tensor rl_next_state = this->rl_env_->make_next_state(
      rl_current_state, chosen_device);
  std::cout << "current state: " << rl_current_state << ", next state: "
      << rl_next_state << ", chosen device:" << chosen_device << "\n";
  torch::Tensor reward = this->rl_env_->calculate_reward(
      chosen_device, rl_current_state);
  this->rl_agent_->append_replay_memory(
      rl_current_state, torch::tensor({float{chosen_device}}, torch::kInt64),
      rl_next_state, torch::tensor({1.f}, torch::kFloat), 0);
  */
#endif
  return true;
}

bool RLTaskMappingPolicy::calc_score_archplacement(
    InnerTask *task, ArchitectureRequirement *arch_placement_req,
    const Mapper &mapper, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
    Score_t *chosen_dev_score,
    const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
        &parray_list,
    std::vector<bool> *is_dev_assigned) {
  return true;
}

bool RLTaskMappingPolicy::calc_score_mdevplacement(
    InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
    const Mapper &mapper,
    std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
    Score_t *average_score,
    const std::vector<
        std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
        &parray_list) {
#if 0
  *average_score = 0;
  const std::vector<std::shared_ptr<SinglePlacementRequirementBase>>
      &placement_reqs_vec = mdev_placement_req->get_placement_reqs_ref();
  member_device_reqs->resize(placement_reqs_vec.size());
  // Task mapper does not allow to map a multi-device task to the same device
  // multiple times. This vector marks an assigned device and filter it
  // out at the next device decision.
  std::vector<bool> is_dev_assigned(
      this->device_manager_->get_num_devices<ParlaDeviceType::All>(), false);
  // Iterate requirements of the devices specified in multi-device placement.
  // All of the member devices should be available.
  for (DevID_t did = 0; did < placement_reqs_vec.size(); ++did) {
    std::shared_ptr<SinglePlacementRequirementBase> placement_req =
        placement_reqs_vec[did];
    std::shared_ptr<DeviceRequirement> dev_req{nullptr};
    Score_t score{0};
    bool is_member_device_available{false};
    if (placement_req->is_dev_req()) {
      dev_req = std::dynamic_pointer_cast<DeviceRequirement>(placement_req);
      DevID_t dev_global_id = dev_req->device()->get_global_id();
      if (!is_dev_assigned[dev_global_id]) {
        is_member_device_available = this->calc_score_devplacement(
            task, dev_req, mapper, &score, parray_list[did]);
        if (is_member_device_available) {
          is_dev_assigned[dev_global_id] = true;
        }
      }
    } else if (placement_req->is_arch_req()) {
      ArchitectureRequirement *arch_req =
          dynamic_cast<ArchitectureRequirement *>(placement_req.get());
      is_member_device_available = this->calc_score_archplacement(
          task, arch_req, mapper, dev_req, &score, parray_list[did],
          &is_dev_assigned);
      if (is_member_device_available) {
        DevID_t dev_global_id = dev_req->device()->get_global_id();
        is_dev_assigned[dev_global_id] = true;
      }
    }
    assert(dev_req != nullptr);
    (*member_device_reqs)[did] = dev_req;
    *average_score += score;
    if (!is_member_device_available) {
      // If any of the device candidates is not available,
      // return false and exclude this option from task mapping.
      return false;
    }
  }
  *average_score /= placement_reqs_vec.size();
  return true;
#endif
}

void RLTaskMappingPolicy::run_task_mapping(
    InnerTask *task, const Mapper &mapper,
    std::vector<std::shared_ptr<DeviceRequirement>> *chosen_devices,
    const std::vector<std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
        &parray_list,
    std::vector<std::shared_ptr<PlacementRequirementBase>>
        *placement_req_options_vec) {
  size_t num_devices = this->device_manager_->get_num_devices();
  std::vector<bool> compatible_devices(num_devices, false);
  std::vector<std::shared_ptr<DeviceRequirement>>
      device_requirements(num_devices);
  for (std::shared_ptr<PlacementRequirementBase> base_req :
       *placement_req_options_vec) {
    if (base_req->is_dev_req()) {
      std::shared_ptr<DeviceRequirement> dev_req =
          std::dynamic_pointer_cast<DeviceRequirement>(base_req);
      const ParlaDevice &device = *(dev_req->device());
      DevID_t global_dev_id = device.get_global_id();
      if (device.check_resource_availability(dev_req.get())) {
        compatible_devices[global_dev_id] = true;
        device_requirements[global_dev_id] = dev_req;
      } else {
        compatible_devices[global_dev_id] = false;
      }
    } else if (base_req->is_arch_req()) {
      // A single architecture placement requirement.
      ArchitectureRequirement *arch_req =
          dynamic_cast<ArchitectureRequirement *>(base_req.get());
      auto placement_options = arch_req->GetDeviceRequirementOptions();
      size_t n_devices = placement_options.size();
      for (size_t d = 0; d < n_devices; ++d) {
        std::shared_ptr<DeviceRequirement> dev_req = placement_options[d];
        const ParlaDevice &device = *(dev_req->device());
        DevID_t global_dev_id = device.get_global_id();
        if (device.check_resource_availability(dev_req.get())) {
          compatible_devices[global_dev_id] = true;
          device_requirements[global_dev_id] = dev_req;
        } else {
          compatible_devices[global_dev_id] = false;
        }
      }
    } else if (base_req->is_multidev_req()) {
      assert(false);
    }
  }

  this->rl_current_state_ =
      this->rl_env_->make_current_state();
  //std::cout << "current state: " << this->rl_current_state_ << ", next state: "
  //    << this->rl_next_state_ << "\n";

  auto [chosen_device_gid, found_device] =
      this->rl_agent_->select_device(
          this->rl_current_state_,
          this->device_manager_->template get_devices<ParlaDeviceType::All>(),
          &compatible_devices);
  if (!found_device || !compatible_devices[chosen_device_gid]) {
    std::cout << "Incompatible or unavailable device was chosen: " << chosen_device_gid << " \n";
    return;
  }
  this->rl_next_state_ = this->rl_env_->make_next_state(
      this->rl_current_state_, chosen_device_gid);
  torch::Tensor reward = this->rl_env_->calculate_reward(
      chosen_device_gid, this->rl_current_state_);
  this->rl_agent_->append_replay_memory(
      this->rl_current_state_,
      torch::tensor({float{chosen_device_gid}}, torch::kInt64),
      this->rl_next_state_, reward);
  this->rl_agent_->optimize_model();
  chosen_devices->clear();
  chosen_devices->push_back(device_requirements[chosen_device_gid]);

  if (task->get_name().find("begin_rl_task") != std::string::npos) {
    this->rl_agent_->incr_episode();
  }
}
