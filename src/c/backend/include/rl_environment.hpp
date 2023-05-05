#ifndef PARLA_RL_ENVIRONMENT
#define PARLA_RL_ENVIRONMENT

#include "device_manager.hpp"

#include <torch/torch.h>

class Mapper;

class RLEnvironment {
public:
  RLEnvironment(DeviceManager *device_manager, Mapper *mapper)
      : device_manager_(device_manager), mapper_(mapper) {}

  torch::Tensor make_current_state();

  torch::Tensor make_next_state(torch::Tensor current_state,
                                DevID_t chosen_device_id);

  torch::Tensor calculate_reward(DevID_t chosen_device_id,
                                 torch::Tensor current_state);
protected:
  DeviceManager *device_manager_;

  Mapper *mapper_;
};

#endif
