#ifndef PARLA_RL_ENVIRONMENT
#define PARLA_RL_ENVIRONMENT

#include "device_manager.hpp"
#include "phases.hpp"

#include <torch/torch.h>

class RLEnvironment {
public:
  RLEnvironment(DeviceManager *device_manager, Mapper *mapper) :
      device_manager_(device_manager), mapper_(mapper) {}

  torch::Tensor calculate_reward(
      DevID_t chosen_device_id, torch::Tensor current_state) {
    if (current_state[0][chosen_device_id].item<int64_t>() == 0) {
      // If the chosen device was idle, give a reward 1.
      return torch::tensor({1.f}, torch::kFloat);
    }
     
    size_t total_running_planned_tasks =
        this->mapper_->atomic_load_total_num_mapped_tasks();
    size_t dev_running_planned_tasks =
        this->mapper_->atomic_load_dev_num_mapped_tasks_device(
            chosen_device_id);
    return torch::tensor(
        {float{1 - (dev_running_planned_tasks / total_running_planned_tasks)}},
        torch::kFloat);
  }

protected:
  DeviceManager *device_manager_;

  Mapper *mapper_;

};

#endif
