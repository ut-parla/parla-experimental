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
      current_state[0][dev_id * 3 + 1] += 1;
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
        current_state[0][dev_id * 3 + 1] += 1;
      }
    }
  }
}

torch::Tensor RLEnvironment::make_current_state(InnerTask *task) {
  DevID_t num_devices =
      this->device_manager_->template get_num_devices(ParlaDeviceType::All);
  torch::Tensor current_state = torch::zeros({1, num_devices * 3}, torch::kDouble);
  double total_nonlocal_bytes{0};
  for (DevID_t d = 0; d < num_devices; ++d) {
    for (size_t pi = 0; pi < task->parray_list[d].size(); ++pi) {
      InnerPArray *parray = task->parray_list[d][pi].first;
      total_nonlocal_bytes += int64_t{parray->get_size()}; 
    }
  }

  for (DevID_t d = 0; d < num_devices; ++d) {
    // TODO: narrowed type conversion.
    int64_t dev_running_planned_tasks =
        this->mapper_->atomic_load_dev_num_mapped_tasks_device(d);
    current_state[0][d * 3] = dev_running_planned_tasks;
    current_state[0][d * 3 + 1] = 0;
    if (total_nonlocal_bytes > 0) {
      double nonlocal_data{0};
      for (size_t pi = 0; pi < task->parray_list[0].size(); ++pi) {
        InnerPArray *parray = task->parray_list[0][pi].first;
        if (!parray_tracker_->get_parray_state(d, parray->parent_id)) {
          nonlocal_data += double{parray->get_size()};
        }
      }
      nonlocal_data /= total_nonlocal_bytes;
      current_state[0][d * 3 + 2] = nonlocal_data; 
    }
  }
  this->make_task_dependency_state(current_state, task);
  return current_state;
}

torch::Tensor RLEnvironment::make_next_state(torch::Tensor current_state,
                              DevID_t chosen_device_id) {
  torch::Tensor next_state = current_state.clone();
  // The first element of a device is the number of task mapped and
  // running on that device. 
  next_state[0][chosen_device_id * 3] += 1;
  return next_state;
}

#if 0
torch::Tensor RLEnvironment::calculate_reward(DevID_t chosen_device_id,
                               InnerTask* task,
                               torch::Tensor current_state) {
  DevID_t num_devices =
      this->device_manager_->template get_num_devices(ParlaDeviceType::All);
  double score = 0;
  double min_nonlocal_data = std::numeric_limits<double>::max();
  DevID_t min_device{0};
  for (DevID_t d = 0; d < num_devices; ++d) {
    double nonlocal_data{0};
    for (size_t pi = 0; pi < task->parray_list[d].size(); ++pi) {
      InnerPArray *parray = task->parray_list[d][pi].first;
      if (!parray_tracker_->get_parray_state(d, parray->parent_id)) {
        nonlocal_data += parray->get_size(); 
      }
    }
    if (min_nonlocal_data > nonlocal_data) {
      min_device = d;
      min_nonlocal_data = nonlocal_data;
    }
  }

  if (min_device == chosen_device_id) {
    score = 1;
  }

  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += score;

  return torch::tensor({score}, torch::kDouble);
}
#endif

torch::Tensor RLEnvironment::calculate_reward(DevID_t chosen_device_id,
                               InnerTask* task,
                               torch::Tensor current_state) {
  double score = 0;
  if (current_state[0][chosen_device_id * 2].item<int64_t>() == 0) {
    // If the chosen device was idle, give a reward 1.
    score = 1.f;
  } else {
    size_t total_running_planned_tasks =
        this->mapper_->atomic_load_total_num_mapped_tasks();
    size_t dev_running_planned_tasks =
        this->mapper_->atomic_load_dev_num_mapped_tasks_device(
            chosen_device_id);
    if (total_running_planned_tasks != 0) {
      score = double{1 - (dev_running_planned_tasks / float(total_running_planned_tasks))};
    }

    DevID_t num_devices =
        this->device_manager_->template get_num_devices(ParlaDeviceType::All);
    double threshold = (1 - (1 / double(num_devices))) - 0.1;
    if (total_running_planned_tasks >= dev_running_planned_tasks) {
      if (score <= threshold) {
        score = -(1 - score);
      }
    }
    std::cout << "score:" << score << ", " <<
      total_running_planned_tasks << ", " << dev_running_planned_tasks << "\n";
  }

  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += score;

  return torch::tensor({score}, torch::kDouble);
}

#if 0
torch::Tensor RLEnvironment::calculate_reward(DevID_t chosen_device_id,
                               torch::Tensor current_state) {
  std::cout << "Calculate reward\n";
  double score = 0;
  if (current_state[0][chosen_device_id * 2].item<int64_t>() == 0) {
    // If the chosen device was idle, give a reward 1.
    score = 1.f;
  }

  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += score;

  std::cout << "Calculate reward:" << score << "\n";
  return torch::tensor({score}, torch::kDouble);
}
#endif


void RLEnvironment::output_reward(size_t episode) {
  std::cout << "Accumulated reward:" << this->reward_accumulation_ << ", and #:" << this->num_reward_accumulation_ << "\n";
  std::ofstream fp("reward.out", std::ios_base::app);
  fp << episode << ", " << (this->reward_accumulation_ / this->num_reward_accumulation_) << "\n";
  this->reward_accumulation_ = 0;
  this->num_reward_accumulation_ = 0;
  fp.close();
}
