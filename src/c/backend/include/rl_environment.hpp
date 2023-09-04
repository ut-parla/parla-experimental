#ifndef PARLA_RL_ENVIRONMENT
#define PARLA_RL_ENVIRONMENT

#include "device_manager.hpp"
#include "rl_utils.h"
#include "runtime.hpp"

#include <torch/torch.h>

#define NUM_TASK_FEATURES 6
#define NUM_DEVICE_FEATURES 7
#define DEVICE_FEATURE_OFFSET NUM_TASK_FEATURES

class RLEnvironment {
public:
  RLEnvironment(DeviceManager *device_manager,
      PArrayTracker *parray_tracker, InnerScheduler *sched)
      : device_manager_(device_manager), parray_tracker_(parray_tracker),
          sched_(sched) {}

  /**
   * @brief Construct a workload state for the current task mapping
   */
  void make_current_workload_state(
      InnerTask *task, torch::Tensor current_state, DevID_t num_devices);
  /**
   * @brief Construct a device-specific state for the current task mapping
   */
  void make_current_device_specific_state(
      InnerTask *task, torch::Tensor current_state, DevID_t num_devices);

  /**
   * @brief Construct a state for the current task mapping
   * (Workload state + device-specific workload)
   */
  torch::Tensor make_current_state(InnerTask *task);

  /**
   * @brief Construct a state after the current task mapping.
   * This does not collect information, but add counters manually based on
   * the task mapping decision.
   */
  torch::Tensor make_next_state(
        torch::Tensor current_state, DevID_t chosen_device_id, InnerTask *task);

  /**
   * @brief Calculate load balancing, data locality, and execution time
   * based score system.
   */
  torch::Tensor calculate_reward(
        DevID_t chosen_device_id, InnerTask* task, torch::Tensor current_state);

  /**
   * @brief Check and return a task type using a task name prefix.
   * If the task type does not exist, create it.
   */
  double check_task_type(InnerTask *task);

  /**
   * @brief Output an accumulated reward in an episode to a file.
   */
  void output_reward(size_t episode);
protected:
  /// Device manager to get device information
  DeviceManager *device_manager_;
  /// PArray tracker to get PArray instance information
  PArrayTracker *parray_tracker_;
  /// Scheduler to get global runtime information
  InnerScheduler *sched_;
  /// The number of reward accumulation to get statistics
  size_t num_reward_accumulation_{0};
  /// Accumulated reward to get statistics
  double reward_accumulation_{0};
  /// Map to manage task type; key is task name or task name prefix
  /// and value is task type number
  std::unordered_map<std::string, double> task_type_map_;
	double last_task_type{0};
};

#endif
