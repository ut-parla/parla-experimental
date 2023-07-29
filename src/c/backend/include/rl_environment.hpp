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

  /// *** Task states + Active dependency/dependent task states per device +
  /// Device states ***
  void make_current_task_state(
      InnerTask *task, torch::Tensor current_state, DevID_t num_devices,
      size_t offset, bool accum);
  void make_current_device_state(
      InnerTask *task, torch::Tensor current_state, DevID_t num_devices);
  void make_current_active_deptask_state(
      InnerTask *task, torch::Tensor current_state, DevID_t num_devices);

  torch::Tensor make_current_state(InnerTask *task);

  torch::Tensor make_next_state(
        torch::Tensor current_state, DevID_t chosen_device_id, InnerTask *task);

  torch::Tensor calculate_reward(DevID_t chosen_device_id,
                                 InnerTask *task,
                                 torch::Tensor current_state, double base_score);

  void output_reward(size_t episode);
protected:
  DeviceManager *device_manager_;
  PArrayTracker *parray_tracker_;
  Mapper *mapper_;

  size_t num_reward_accumulation_{0};
  double reward_accumulation_{0};

};

#endif
