#ifndef PARLA_RL_ENVIRONMENT
#define PARLA_RL_ENVIRONMENT

#include "device_manager.hpp"
#include "runtime.hpp"

#include <torch/torch.h>

class Mapper;

class RLEnvironment {
public:
  RLEnvironment(DeviceManager *device_manager,
      PArrayTracker *parray_tracker, Mapper *mapper)
      : device_manager_(device_manager), parray_tracker_(parray_tracker),
          mapper_(mapper) {}

  void make_task_dependency_state(torch::Tensor current_state, InnerTask *task);

  torch::Tensor make_current_state(InnerTask *task);

  torch::Tensor make_next_state(torch::Tensor current_state,
                                DevID_t chosen_device_id);

  torch::Tensor calculate_reward(DevID_t chosen_device_id,
                                 InnerTask *task,
                                 torch::Tensor current_state);

  void output_reward(size_t episode);
protected:
  DeviceManager *device_manager_;
  PArrayTracker *parray_tracker_;
  Mapper *mapper_;

  size_t num_reward_accumulation_{0};
  double reward_accumulation_{0};

};

#endif
