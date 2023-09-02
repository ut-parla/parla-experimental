#ifndef PARLA_RL_ENVIRONMENT
#define PARLA_RL_ENVIRONMENT

#include "device_manager.hpp"
#include "rl_utils.h"
#include "runtime.hpp"

#include <torch/torch.h>

#define NUM_TASK_FEATURES 7
#define NUM_DEP_TASK_FEATURES 0
#define NUM_DEVICE_FEATURES 8
#define DEVICE_FEATURE_OFFSET (NUM_TASK_FEATURES + NUM_DEP_TASK_FEATURES * 2)

class RLEnvironment {
public:
  RLEnvironment(DeviceManager *device_manager,
      PArrayTracker *parray_tracker, InnerScheduler *sched)
      : device_manager_(device_manager), parray_tracker_(parray_tracker),
          sched_(sched) {}

  /// *** Task states + Active dependency/dependent task states per device +
  /// Device states ***
  void make_current_task_state(
      InnerTask *task, torch::Tensor current_state, DevID_t num_devices);
  void make_current_device_state(
      InnerTask *task, torch::Tensor current_state, DevID_t num_devices);

  torch::Tensor make_current_state(InnerTask *task);

  torch::Tensor make_next_state(
        torch::Tensor current_state, DevID_t chosen_device_id, InnerTask *task);

  torch::Tensor calculate_reward(DevID_t chosen_device_id,
                                 InnerTask *task,
                                 torch::Tensor current_state, double base_score);
  torch::Tensor calculate_reward2(DevID_t chosen_device_id,
                                  InnerTask* task,
                                  torch::Tensor current_state);
  torch::Tensor calculate_reward_loadbalancing(DevID_t chosen_device_id,
                                 InnerTask *task,
                                 torch::Tensor current_state);
  torch::Tensor calculate_reward_parla(
                               DevID_t chosen_device_id,
                               InnerTask* task,
                               torch::Tensor current_state);

  double check_task_type(InnerTask *task);
  double check_task_type_using_name(InnerTask *task);

  void output_reward(size_t episode);
protected:
  DeviceManager *device_manager_;
  PArrayTracker *parray_tracker_;
  InnerScheduler *sched_;

  size_t num_reward_accumulation_{0};
  double reward_accumulation_{0};

  std::unordered_map<std::string, double> task_type_map_;
  std::unordered_map<std::string, std::pair<double, size_t>> task_compltime_delta_map_;
	double last_task_type{0};
  // Accumulated task completion time to keep track of
  // average of the completion time among tasks.
  double total_task_completiontime{0};
  size_t num_task_completiontime{0};
};

#endif
