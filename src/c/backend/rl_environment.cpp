#include "include/rl_environment.hpp"
#include "include/phases.hpp"

torch::Tensor RLEnvironment::make_current_state() {
  DevID_t num_devices =
      this->device_manager_->template get_num_devices(ParlaDeviceType::All);
  torch::Tensor current_state = torch::zeros({1, num_devices});
  for (DevID_t d = 0; d < num_devices; ++d) {
    // TODO: narrowed type conversion.
    int64_t dev_running_planned_tasks =
        this->mapper_->atomic_load_dev_num_mapped_tasks_device(d);
    current_state[0][d] = dev_running_planned_tasks;
  }
  return current_state;
}

torch::Tensor RLEnvironment::make_next_state(torch::Tensor current_state,
                              DevID_t chosen_device_id) {
  torch::Tensor next_state = current_state.clone();
  next_state[0][chosen_device_id] += 1;
  return next_state;
}

torch::Tensor RLEnvironment::calculate_reward(DevID_t chosen_device_id,
                               torch::Tensor current_state) {
  if (current_state[0][chosen_device_id].item<int64_t>() == 0) {
    // If the chosen device was idle, give a reward 1.
    return torch::tensor({1.f}, torch::kFloat);
  }

  size_t total_running_planned_tasks =
      this->mapper_->atomic_load_total_num_mapped_tasks();
  size_t dev_running_planned_tasks =
      this->mapper_->atomic_load_dev_num_mapped_tasks_device(
          chosen_device_id);
  double score = 0;
  if (total_running_planned_tasks != 0) {
    score = double{1 - (dev_running_planned_tasks / float(total_running_planned_tasks))};
  }

  DevID_t num_devices =
      this->device_manager_->template get_num_devices(ParlaDeviceType::All);
  double threshold = (1 - (1 / double(num_devices))) - 0.1;
  if (total_running_planned_tasks >= dev_running_planned_tasks) {
    if (score <= threshold) {
      score = -(1 + score);
    }
  }

  std::cout << "score:" << score << ", " <<
    total_running_planned_tasks << ", " << dev_running_planned_tasks << "\n";

  return torch::tensor({score}, torch::kDouble);
}

