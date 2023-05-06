#include "include/rl_task_mapper.hpp"

RLTaskMappingPolicy::RLTaskMappingPolicy(
    DeviceManager *device_manager, PArrayTracker *parray_tracker,
    Mapper *mapper)
    : MappingPolicy(device_manager, parray_tracker) {
  size_t num_devices = device_manager->get_num_devices();
  this->rl_agent_ = new RLAgent(num_devices, num_devices);
  this->rl_env_ = new RLEnvironment(this->device_manager_, mapper);
}

bool RLTaskMappingPolicy::calc_score_devplacement(
    InnerTask *task,
    const std::shared_ptr<DeviceRequirement> &dev_placement_req,
    const Mapper &mapper, Score_t *score,
    const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
              &parray_list) {
  torch::Tensor rl_current_state = this->rl_env_->make_current_state();
  uint32_t chosen_device = this->rl_agent_->select_device(
      rl_current_state,
      this->device_manager_->template get_devices<ParlaDeviceType::All>());
  torch::Tensor rl_next_state = this->rl_env_->make_next_state(
      rl_current_state, chosen_device);
  std::cout << "current state: " << rl_current_state << ", next state: "
      << rl_next_state << ", chosen device:" << chosen_device << "\n";
  torch::Tensor reward = this->rl_env_->calculate_reward(
      chosen_device, rl_current_state);
  this->rl_agent_->append_replay_memory(
      rl_current_state, torch::tensor({float{chosen_device}}, torch::kInt64),
      rl_next_state, torch::tensor({1.f}, torch::kFloat), 0);
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
  return true;
}
