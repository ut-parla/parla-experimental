#include "include/rl_environment.hpp"
#include "include/phases.hpp"

void RLEnvironment::make_task_dependency_state(torch::Tensor current_state,
    InnerTask *task) {
  if (task->name.find("global_0") != std::string::npos) {
    return;
  }

  DevID_t num_devices =
      this->device_manager_->template get_num_devices(ParlaDeviceType::All);
  std::vector<std::unordered_set<uint64_t>> task_id_map(num_devices);
  for (size_t i = 0; i < task->dependencies.size(); ++i) {
    InnerTask *dependency = task->dependencies.at(i);
    if (dependency->is_data.load()) {
      continue;
    }
    uint64_t taskid = dependency->id;
    for (ParlaDevice *mapped_device : dependency->get_assigned_devices()) {
      DevID_t dev_id = mapped_device->get_global_id();
      auto got = task_id_map[dev_id].find(taskid);
      if (got != task_id_map[dev_id].end()) {
        continue;  
      }
      std::cout << task->name << "'s dependency:" << dependency->name << " on device: " << dev_id << "\n";
      task_id_map[dev_id].emplace(taskid);
      current_state[0][dev_id * 2 + 1] += 1;
    }
  }

  for (size_t i = 0; i < task->dependents.size(); ++i) {
    InnerTask *dependent = task->dependents.at(i);
    std::cout << task->name << "'s depenent:" <<
      dependent->name << "\n";
    if (dependent->name.find("global_0") != std::string::npos) {
      continue;
    }

    for (size_t j = 0; j < dependent->dependencies.size(); ++j) {
      InnerTask *dependent_dependency = dependent->dependencies.at(j);
      if (dependent_dependency->is_data.load()) {
        continue;
      }

      uint64_t taskid = dependent_dependency->id;
      for (ParlaDevice *mapped_device :
          dependent_dependency->get_assigned_devices()) {
        DevID_t dev_id = mapped_device->get_global_id();
        auto got = task_id_map[dev_id].find(taskid);
        if (got != task_id_map[dev_id].end()) {
          continue;  
        }
        task_id_map[dev_id].emplace(taskid);
        std::cout << task->name << "'s depenent's dependency:" <<
          dependent_dependency->name << " on device: " << dev_id << "\n";
        current_state[0][dev_id * 2 + 1] += 1;
      }
    }
  }
}

torch::Tensor RLEnvironment::make_current_state(InnerTask *task) {
  DevID_t num_devices =
      this->device_manager_->template get_num_devices(ParlaDeviceType::All);
  torch::Tensor current_state = torch::zeros({1, num_devices * 2});
  for (DevID_t d = 0; d < num_devices; ++d) {
    // TODO: narrowed type conversion.
    double dev_running_planned_task_loads =
        this->mapper_->atomic_load_dev_num_mapped_tasks_device(d);
    std::cout << d << "'s load:" << dev_running_planned_task_loads << "\n";
    current_state[0][d * 2] = dev_running_planned_task_loads;
    current_state[0][d * 2 + 1] = 0;
  }
  this->make_task_dependency_state(current_state, task);
  return current_state;
}

torch::Tensor RLEnvironment::make_next_state(torch::Tensor current_state,
                              DevID_t chosen_device_id) {
  torch::Tensor next_state = current_state.clone();
  // The first element of a device is the number of task mapped and
  // running on that device. 
  next_state[0][chosen_device_id * 2] += 1;
  return next_state;
}

/// Calculate load balancing reward at the mapping phase.
torch::Tensor RLEnvironment::calculate_loadbalancing_reward(
    DevID_t chosen_device_id, torch::Tensor current_state) {
  double score = 0;
  if (current_state[0][chosen_device_id * 2].item<int64_t>() == 0) {
    // If the chosen device has been idle, give a reward 1.
    score = 1.f;
  } else {
    double total_running_planned_tasks =
        this->mapper_->atomic_load_total_num_mapped_tasks();
    double dev_running_planned_tasks =
        this->mapper_->atomic_load_dev_num_mapped_tasks_device(
            chosen_device_id);
    if (total_running_planned_tasks != 0) {
      // Apply root for smooth gradient descent.
      score = double{1 -
          pow(dev_running_planned_tasks / total_running_planned_tasks, 0.5)};
    }

    DevID_t num_devices =
        this->device_manager_->template get_num_devices(ParlaDeviceType::All);
    // If too many tasks are skewed to a device, that choice was not good.
    // So, give a negative reward.
    double threshold = (1 - pow(1 / double(num_devices), 0.5)) - 0.05;
    if (score <= threshold) {
      score = -(1 - score);
    }
  }

  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += score;
  std::cout << "Load balancing score:" << score << "\n";
  return torch::tensor({score}, torch::kDouble);
}

#if 0
torch::Tensor RLEnvironment::calculate_loadbalancing_reward(DevID_t chosen_device_id,
                               torch::Tensor current_state) {
  double score = -1.f;
  if (current_state[0][chosen_device_id * 2].item<int64_t>() == 0) {
    // If the chosen device was idle, give a reward 1.
    score = 1.f;
  } else {
    size_t total_running_planned_tasks =
        this->mapper_->atomic_load_total_num_mapped_tasks();
    size_t dev_running_planned_tasks =
        this->mapper_->atomic_load_dev_num_mapped_tasks_device(
            chosen_device_id);
    double load_ratio = dev_running_planned_tasks / float(total_running_planned_tasks);
    DevID_t num_devices =
        this->device_manager_->template get_num_devices(ParlaDeviceType::All);
    double threshold = 1 / float(num_devices);
    ioverride f (load_ratio <= threshold) {
      score = 1.f;
    }
    /*
    std::cout << "score:" << score << ", " <<
      total_running_planned_tasks << ", " << dev_running_planned_tasks << "\n";
    */
  }

  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += score;
  std::cout << "Total score:" << score << "\n";


  /*
  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += score;
  */

  return torch::tensor({score}, torch::kDouble);
}
#endif

void RLEnvironment::output_reward(size_t episode) {
  std::cout << "Accumulated reward:" << this->reward_accumulation_ << ", and #:" << this->num_reward_accumulation_ << "\n";
  std::ofstream fp("reward.out", std::ios_base::app);
  fp << episode << ", " << this->reward_accumulation_ << ", "
    << this->num_reward_accumulation_ << ", " << this->loadbalancing_reward_accumulation_ <<
    ", " << this->waittime_reward_accumulation_ << ", " <<
    (this->reward_accumulation_ / this->num_reward_accumulation_) << "\n";
  this->loadbalancing_reward_accumulation_ = 0;
  this->waittime_reward_accumulation_ = 0;
  this->reward_accumulation_ = 0;
  this->num_reward_accumulation_ = 0;
  fp.close();
}

/// Calculate the final reward when a task is about to be launched.
/// The final score is weighted by wait time from resource reservation phase.
/// If the task was launched immediately after resource reservation,
/// this weight is 1, which im
/// it implies that the task is not blocked by any dependency, 
torch::Tensor RLEnvironment::calculate_reward(
    InnerTask* task, DevID_t chosen_device_id,
    torch::Tensor base_score,
    double dev_accum_idletime_mapping,
    double dev_accum_idletime_launching) {
  double idletime_score{1};
    std::cout << "launching idle time:" << dev_accum_idletime_launching <<
      ", mapping idle time:" << dev_accum_idletime_mapping << "\n";

  if (dev_accum_idletime_launching > 0) {
    idletime_score = pow(double{1} -
      (dev_accum_idletime_launching - dev_accum_idletime_mapping) /
          dev_accum_idletime_launching, 0.5);
  }
  double total_score = (idletime_score * base_score.item<float>());

  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += total_score;
  this->waittime_reward_accumulation_ += idletime_score;
  this->loadbalancing_reward_accumulation_ += base_score.item<float>();
  std::cout << " Base score:" << base_score.item<float>() << "\n";
  std::cout << " Wait-time score:" << idletime_score << "\n";
  std::cout << "Total score:" << total_score << "\n";
  return torch::tensor({total_score}, torch::kDouble);
}
