#include "include/rl_task_mapper.hpp"

RLTaskMappingPolicy::RLTaskMappingPolicy(
    DeviceManager *device_manager, PArrayTracker *parray_tracker,
    InnerScheduler *scheduler, bool is_training_mode)
    : MappingPolicy(device_manager, parray_tracker) {
  size_t num_devices = device_manager->get_num_devices();
  this->rl_agent_ = new RLAgent(NUM_TASK_FEATURES +
      NUM_DEP_TASK_FEATURES * 2 + num_devices * NUM_DEVICE_FEATURES,
      num_devices, num_devices, is_training_mode);
  this->rl_env_ = new RLEnvironment(this->device_manager_, parray_tracker, scheduler);
}

RLTaskMappingPolicy::~RLTaskMappingPolicy() {
  delete this->rl_agent_;
  delete this->rl_env_;
}

bool RLTaskMappingPolicy::calc_score_devplacement(
    InnerTask *task,
    const std::shared_ptr<DeviceRequirement> &dev_placement_req,
    InnerScheduler *scheduler, Score_t *score,
    const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
              &parray_list) {
  return true;
}

bool RLTaskMappingPolicy::calc_score_archplacement(
    InnerTask *task, ArchitectureRequirement *arch_placement_req,
    InnerScheduler *scheduler, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
    Score_t *chosen_dev_score,
    const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
        &parray_list,
    std::vector<bool> *is_dev_assigned) {
  return true;
}

bool RLTaskMappingPolicy::calc_score_mdevplacement(
    InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
    InnerScheduler *scheduler,
    std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
    Score_t *average_score,
    const std::vector<
        std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
        &parray_list) {}

void RLTaskMappingPolicy::run_task_mapping(
    InnerTask *task, InnerScheduler *scheduler,
    std::vector<std::shared_ptr<DeviceRequirement>> *chosen_devices,
    const std::vector<std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
        &parray_list,
    std::vector<std::shared_ptr<PlacementRequirementBase>>
        *placement_req_options_vec) {
  size_t num_devices = this->device_manager_->get_num_devices();
  std::vector<bool> compatible_devices(num_devices, false);
  std::vector<std::shared_ptr<DeviceRequirement>>
      device_requirements(num_devices);
  size_t num_compatible_devices{0};
  DevID_t chosen_device_gid{0};
  for (std::shared_ptr<PlacementRequirementBase> base_req :
       *placement_req_options_vec) {
    if (base_req->is_dev_req()) {
      std::shared_ptr<DeviceRequirement> dev_req =
          std::dynamic_pointer_cast<DeviceRequirement>(base_req);
      const ParlaDevice &device = *(dev_req->device());
      DevID_t global_dev_id = device.get_global_id();
      // Store this to immediately choose this device
      // if a single device is compatible.
      chosen_device_gid = global_dev_id;
      device_requirements[global_dev_id] = dev_req;
      compatible_devices[global_dev_id] = true;
      ++num_compatible_devices;
    } else if (base_req->is_arch_req()) {
      // A single architecture placement requirement.
      ArchitectureRequirement *arch_req =
          dynamic_cast<ArchitectureRequirement *>(base_req.get());
      auto placement_options = arch_req->GetDeviceRequirementOptions();
      size_t n_devices = placement_options.size();
      for (size_t k = 0; k < n_devices; ++k) {
        std::shared_ptr<DeviceRequirement> dev_req = placement_options[k];
        ParlaDevice *device = dev_req->device();
        DevID_t global_dev_id = device->get_global_id();
        compatible_devices[global_dev_id] = true;
        // Store this to immediately choose this device
        // if a single device is compatible.
        chosen_device_gid = global_dev_id;
        device_requirements[global_dev_id] = dev_req;
        ++num_compatible_devices;
      }
    } else if (base_req->is_multidev_req()) {
      assert(false);
    }
  }

  chosen_devices->clear();
  if (num_compatible_devices == 1) {
    chosen_devices->push_back(device_requirements[chosen_device_gid]);
  } else {
    this->rl_current_state_ =
        this->rl_env_->make_current_state(task);
    DevID_t chosen_device_gid =
        this->rl_agent_->select_device(
            this->rl_current_state_,
            this->device_manager_->template get_devices<ParlaDeviceType::All>(),
            &compatible_devices);
    chosen_devices->push_back(device_requirements[chosen_device_gid]);
    if (this->rl_agent_->is_training_mode()) {
      if (check_valid_tasks(task->name)) {
        this->rl_next_state_ = this->rl_env_->make_next_state(
            this->rl_current_state_, chosen_device_gid, task);
        torch::Tensor reward = this->rl_env_->calculate_reward(
            chosen_device_gid, task, this->rl_current_state_);
        this->rl_agent_->append_mapped_task_info(
            task, this->rl_current_state_, this->rl_next_state_,
            torch::tensor({float{chosen_device_gid}}, torch::kInt64),
            reward);
      }
    }

    if (task->get_name().find("begin_rl_task") != std::string::npos) {
      this->device_manager_->reset_device_timers();
    }
  }
}
