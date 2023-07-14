#ifndef PARLA_RL_ENVIRONMENT
#define PARLA_RL_ENVIRONMENT

#include "device_manager.hpp"
#include "runtime.hpp"

#include <torch/torch.h>

class Mapper;

class RLEnvironment {
public:
  RLEnvironment(DeviceManager *device_manager, Mapper *mapper)
      : device_manager_(device_manager), mapper_(mapper) {}

  void make_task_dependency_state(torch::Tensor current_state, InnerTask *task);

  torch::Tensor make_current_state(InnerTask *task);

  torch::Tensor make_next_state(torch::Tensor current_state,
                                DevID_t chosen_device_id);

  torch::Tensor calculate_loadbalancing_reward(DevID_t chosen_device_id,
                                 torch::Tensor current_state);

  torch::Tensor calculate_reward(InnerTask* task, DevID_t chosen_device_id,
                                 torch::Tensor current_state,
                                 double dev_accum_idletime_mapping,
                                 double dev_accum_idletime_launching);

  void output_reward(size_t episode);
protected:
  DeviceManager *device_manager_;

  Mapper *mapper_;

  size_t num_reward_accumulation_{0};
  double reward_accumulation_{0};
  double waittime_reward_accumulation_{0};
  double loadbalancing_reward_accumulation_{0};

};

#endif
