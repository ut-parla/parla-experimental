#include "include/rl_environment.hpp"
#include "include/phases.hpp"
#include "include/runtime.hpp"
#include "include/profiling.hpp"

#include <chrono>
#include <cmath>

#define PRINT_LOG false 

using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

double RLEnvironment::check_task_type(InnerTask *task) {
  double task_type;
  std::string task_type_str = task->name.substr(0, task->name.find("_"));
  auto found = this->task_type_map_.find(task_type_str);
  if (found == this->task_type_map_.end()) {
    task_type = this->last_task_type;
    //std::cout << task_type_str << "'s task type found \n";
    this->task_type_map_[task_type_str] = this->last_task_type++;
  } else {
    task_type = found->second;
  }
  //std::cout << task_type_str << "'s task type:" << task_type << "\n";
  return task_type;
}

double RLEnvironment::check_task_type_using_name(InnerTask *task) {
  double task_type;
  auto found = this->task_type_map_.find(task->name);
  if (found == this->task_type_map_.end()) {
    task_type = this->last_task_type;
    //std::cout << task_type_str << "'s task type found \n";
    this->task_type_map_[task->name] = this->last_task_type++;
  } else {
    task_type = found->second;
  }
  //std::cout << task_type_str << "'s task type:" << task_type << "\n";
  return task_type;
}

void RLEnvironment::make_current_task_state(
    InnerTask *task, torch::Tensor current_state, DevID_t num_devices) {
  // 1) # of active dependencies:
  uint32_t num_considering_task_states{
      Task::State::RUNNING - Task::State::MAPPED + 1};
  // This vector counts the number of dependencies (parents)
  // which are between in MAPPED and RUNNING states.
  // Those dependencies typically could affect the current target
  // task's execution.
  std::vector<int64_t> num_dependencies(num_considering_task_states, 0);
  for (size_t i = 0; i < task->dependencies.size(); ++i) {
    // Only consider computation tasks, not data movement tasks since
    // those tasks are considered through data locality related states.
    InnerTask *dependency = task->dependencies.at(i);
    if (!check_valid_tasks(dependency->name)) { continue; }
    if (dependency->is_data.load()) { continue; }
    Task::State task_state = task->get_state();
    if (task_state < Task::State::MAPPED ||
        task_state > Task::State::RUNNING) { continue; }
    ++num_dependencies[task_state - Task::State::MAPPED];
  }

  // 2) Total bytes of IN/INOUT/OUT PArrays.
  // XXX(hc): This assumes a single device task; so PArrays are initialized at
  // 0th device.
  double total_bytes{0};
  for (size_t pi = 0; pi < task->parray_list[0].size(); ++pi) {
    InnerPArray *parray = task->parray_list[0][pi].first;
    total_bytes += int64_t{parray->get_size()}; 
  }

  // TODO(hc): These states are not normalized yet.
  // For now, rely on torch::normalize layers.
  size_t offset{0};
  for (uint32_t i = 0; i < num_considering_task_states; ++i, ++offset) {
    current_state[0][i] = num_dependencies[i];
  }
  current_state[0][offset++] = (total_bytes / double{1 << 20});
  // 3) Task type.
  double task_type = this->check_task_type(task);
  current_state[0][offset] = task_type;
#if PRINT_LOG
  LOG_INFO("Debug", "Current Task State Print");
  size_t print_offset{0};
  for (uint32_t i = 0; i < num_considering_task_states; ++i, ++print_offset) {
    LOG_INFO("Debug", "current state[{}]={}",
        i, current_state[0][i].item<double>());
  }
  LOG_INFO("Debug", "current state[{}]={}",
      print_offset, current_state[0][print_offset].item<double>());
  ++print_offset;
  LOG_INFO("Debug", "current state[{}]={}",
      print_offset, current_state[0][print_offset].item<double>());
  LOG_INFO("Debug", "Current Task State Print [done]");
#endif
}

void RLEnvironment::make_current_device_state(
    InnerTask *task, torch::Tensor current_state, DevID_t num_devices) {
  // 1) The number of tasks for each state (TODO(hc): for now ignore CPU).
  double tnmt = this->sched_->atomic_load_total_num_mapped_tasks();
  size_t nmt_m = 0, nmt_rs = 0, nmt_rd = 0, nmt_rn = 0;
  size_t num_considering_task_states{4};
  size_t offset{DEVICE_FEATURE_OFFSET + NUM_DEVICE_FEATURES};
  if (tnmt == 0) {
    for (DevID_t d = 1; d < num_devices; ++d) {
      current_state[0][offset] = 0;
      current_state[0][offset+1] = 0;
      current_state[0][offset+2] = 0;
      current_state[0][offset+3] = 0;
      offset += NUM_DEVICE_FEATURES;
    }
  } else {
    for (DevID_t d = 1; d < num_devices; ++d) {
      nmt_m = this->sched_->atomic_load_dev_num_tasks_mapped_states(d); 
      nmt_rs = this->sched_->atomic_load_dev_num_tasks_resreserved_states(d);
      nmt_rd = this->sched_->atomic_load_dev_num_ready_tasks(d);
      nmt_rn = this->sched_->atomic_load_dev_num_running_tasks(d);
      current_state[0][offset] = nmt_m / tnmt;
      current_state[0][offset+1] = nmt_rs / tnmt;
      current_state[0][offset+2] = nmt_rd / tnmt;
      current_state[0][offset+3] = nmt_rn / tnmt;
      offset += NUM_DEVICE_FEATURES;
    }
  }

#if PRINT_LOG
  size_t print_offset{DEVICE_FEATURE_OFFSET + NUM_DEVICE_FEATURES};
  for (DevID_t d = 1; d < num_devices; ++d) {
    for (uint32_t st = 0; st < num_considering_task_states; ++st) {
      LOG_INFO("Debug", "current state[{}]={}", print_offset + st,
          current_state[0][print_offset + st].item<double>());
    }
    print_offset += NUM_DEVICE_FEATURES;
  }
#endif

  // 2) Data locality: Calculate the total IN/OUT/INOUT PArrays' bytes
  double total_bytes{0}, local_bytes{0};
  // XXX(hc): This assumes a single device task; so PArrays are initialized at
  // 0th device.
  for (size_t pi = 0; pi < task->parray_list[0].size(); ++pi) {
    InnerPArray *parray = task->parray_list[0][pi].first;
    total_bytes += double{parray->get_size()};
  }

  // TODO(hc): ignore CPU feature.
  for (DevID_t d = 1; d < num_devices; ++d) {
    double nonlocal_bytes{0};
    offset =
      DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES +
      num_considering_task_states;
    if (total_bytes > 0) {
      for (size_t pi = 0; pi < task->parray_list[0].size(); ++pi) {
        InnerPArray *parray = task->parray_list[0][pi].first;
        if (!this->parray_tracker_->get_parray_state(d, parray->parent_id)) {
          nonlocal_bytes += double{parray->get_size()};  
        }
      }
      current_state[0][offset] = nonlocal_bytes / total_bytes;
    } else {
      current_state[0][offset] = 0;
    }
    ++offset;

#if PRINT_LOG
    LOG_INFO("Debug", "current state[{}]={}",
        offset - 1, current_state[0][offset - 1].item<double>());
#endif

    // 3) Reservable memory.
    ParlaDevice* device =
        this->device_manager_->get_device_by_global_id(d);
    const ResourcePool_t &device_pool = device->get_resource_pool();
    const ResourcePool_t &reserved_device_pool = device->get_reserved_pool();  
    int64_t total_device_memory = device_pool.get(Resource::Memory);
    int64_t remaining_device_memory =
        reserved_device_pool.get(Resource::Memory);
    current_state[0][offset] =
      (total_device_memory > 0)?
          (remaining_device_memory / double{total_device_memory}) : 0;
#if PRINT_LOG
    LOG_INFO("Debug", "current state[{}]={}",
        offset, current_state[0][offset].item<double>());
#endif
  }
}

torch::Tensor RLEnvironment::make_current_state(InnerTask *task) {
  DevID_t num_devices =
      this->device_manager_->template get_num_devices(ParlaDeviceType::All);
  torch::Tensor current_state =
      torch::zeros({1,
          NUM_TASK_FEATURES + (NUM_DEP_TASK_FEATURES * 2) +
          (num_devices * NUM_DEVICE_FEATURES)}, torch::kDouble);
  this->make_current_task_state(task, current_state, num_devices);
  this->make_current_device_state(task, current_state, num_devices);

  return current_state;
}

torch::Tensor RLEnvironment::make_next_state(
    torch::Tensor current_state, DevID_t chosen_device_id, InnerTask *task) {
  torch::Tensor next_state = current_state.clone();
  ParlaDevice* device =
      this->device_manager_->get_device_by_global_id(chosen_device_id);
  // The first element of a device is the number of task mapped and
  // running on that device. 
  next_state[0][
      DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES] += 1;

  // Reservable memory.
  const ResourcePool_t &device_pool = device->get_resource_pool();
  ResourcePool_t &task_pool = task->device_constraints[chosen_device_id];
  const ResourcePool_t &reserved_device_pool = device->get_reserved_pool();  
  int64_t task_memory = task_pool.get(Resource::Memory);
  int64_t total_memory = device_pool.get(Resource::Memory);
  int64_t remaining_memory =
      reserved_device_pool.get(Resource::Memory) - task_memory;

  current_state[0][
      DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES] += 1;

  if (task->name.find("gemm1") != std::string::npos) {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 4] += 1;
  } else if (task->name.find("subcholesky") != std::string::npos) {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 5] += 1;
  } else if (task->name.find("gemm2") != std::string::npos) {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 6] += 1;
  } else if (task->name.find("solve") != std::string::npos) {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 7] += 1;
  } else {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 8] += 1;
  }

  next_state[0][
      DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 1] = 0;
  next_state[0][
      DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 2] =
      (total_memory > 0)? (remaining_memory / double{total_memory}) : 0;

#if 0
  if (task->name.find("gemm1") != std::string::npos) {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 4] += 1;
  }
  else if (task->name.find("subcholesky") != std::string::npos) {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 5] += 1;
  }
  else if (task->name.find("gemm2") != std::string::npos) {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 6] += 1;
  }
  else if (task->name.find("solve") != std::string::npos) {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 7] += 1;
  } else {
    current_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 8] += 1;
  }
#endif

#if 0
  // Average idle and non-idle time features.
	TimePoint current_time_point = std::chrono::system_clock::now();
  auto [idle_time, nonidle_time] = device->get_total_idle_time(current_time_point);
  double current_time = device->current_timepoint_count_from_beginning();
  if (current_time > 0) {
    next_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 3] =
        idle_time / current_time;
    next_state[0][
        DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 4] =
        nonidle_time/ current_time;
  }
#endif

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
                               torch::Tensor current_state, double base_score) {
  double score{0};
  if (chosen_device_id == 0) {
#if PRINT_LOG
    std::cout << task->name << " selected CPU 0\n";
#endif
    score = 0.f;
  } else {
    double completion_time_epochs = task->completion_time_epochs;
    if (completion_time_epochs == 0) {
      double average = this->total_task_completiontime / this->num_task_completiontime;
      if (average == 0) {
        score = 1;
      } else {
        score = average / 10;
      }
    } else {
      double interval = task->completion_time_epochs - task->max_depcompl_time_epochs;
      this->total_task_completiontime += interval;
      this->num_task_completiontime += 1;
      double average = this->total_task_completiontime / this->num_task_completiontime;
      double norm_interval = average / (10 * interval);
      //score = 1 / interval;
      score = norm_interval;
#if PRINT_LOG
      std::cout << "max decompl time epochs:" << interval << " average:" << average << " score:" << score << "\n";
#endif
    }

    ++this->num_reward_accumulation_;
    this->reward_accumulation_ += score;
  }
  return torch::tensor({score}, torch::kDouble);
}

torch::Tensor RLEnvironment::calculate_reward2(DevID_t chosen_device_id,
                               InnerTask* task,
                               torch::Tensor current_state) {
  double score{0};
  if (chosen_device_id == 0) {
#if PRINT_LOG
    std::cout << task->name << " selected CPU 0\n";
#endif
    score = 0.f;
  } else {
    std::string task_name = task->name;
    uint32_t num_devices = this->device_manager_->template get_num_devices(ParlaDeviceType::CUDA);
    /*
    size_t num_siblings = task->approximated_num_siblings;
    std::cout << "num siblings:" << num_siblings << "\n";
    size_t weight_delta = ceil(num_siblings / num_devices);
    std::cout << "weight delta:" << weight_delta << "\n";
    weight_delta = (weight_delta < 1)? 1 : weight_delta;
    */

    /*
    double delta = (num_siblings == 0)?
        (task->completion_time_epochs - task->max_depcompl_time_epochs) :
        (task->completion_time_epochs - task->max_depcompl_time_epochs) /
            double{(num_siblings / num_devices)};
    */
    double delta =
        (task->completion_time_epochs - task->max_depcompl_time_epochs);
    double old_delta{0};
    auto found = this->task_compltime_delta_map_.find(task_name);
    if (found != this->task_compltime_delta_map_.end()) {
      old_delta = (found->second.first / found->second.second);
      if (old_delta > 0.8 * delta) {
        score = 1;
      } else if (0.8 * delta > old_delta) {
        score = -1;
      }
      found->second.first += delta;
      found->second.second += 1;
    } else {
      this->task_compltime_delta_map_[task_name] = {delta, 1};
    }

    std::cout << task_name << " chosen dev:" << chosen_device_id <<
      " orig delta " << delta << 
      " vs old delta " << old_delta << " score:" << score << "\n";
    log_rl_msg(2, "calc_reward,"+task->name+", "+
        std::to_string(chosen_device_id)+", "+
        std::to_string(task->completion_time_epochs)+", "+
        std::to_string(task->max_depcompl_time_epochs)+", "+
        std::to_string(delta));
  }
  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += score;
  return torch::tensor({score}, torch::kDouble);
}


#if 0
torch::Tensor RLEnvironment::calculate_reward(DevID_t chosen_device_id,
                               InnerTask* task,
                               torch::Tensor current_state, double base_score) {
  double score{0};
  if (chosen_device_id == 0) {
    std::cout << task->name << " selected CPU 0\n";
    score = 0.f;
  } else {
    if (base_score > 0) {
      // If the chosen device was idle, give a reward 1.
      score = 1.f;
    } else {
      DevID_t num_devices =
          this->device_manager_->template get_num_devices(ParlaDeviceType::CUDA);
      DevID_t best_device{chosen_device_id};
      double max_idle_time{0};
      // Only consider CUDA devices, and their index starts from 1.
			TimePoint current_time_point = std::chrono::system_clock::now();
      for (DevID_t d = 1; d < num_devices + 1; ++d) {
        ParlaDevice* device = this->device_manager_->get_device_by_global_id(d);
        auto [idle_time, nonidle_time] = device->get_total_idle_time(current_time_point);
        if (max_idle_time < idle_time) {
          max_idle_time = idle_time;
          best_device = d;
        }
      }

      if (chosen_device_id == best_device) {
        score = double{1};
      } else {
        score = 0;
      }

      std::cout << "chosen device:" << chosen_device_id << ", score:" << score << "\n";
    }

    ++this->num_reward_accumulation_;
    this->reward_accumulation_ += score;
  }
  return torch::tensor({score}, torch::kDouble);
}
#endif

#if 0
ORIGINAL
torch::Tensor RLEnvironment::calculate_reward(DevID_t chosen_device_id,
                               InnerTask* task,
                               torch::Tensor current_state, double base_score) {
  double score{0};
  if (chosen_device_id == 0) {
    std::cout << task->name << " selected CPU 0\n";
    score = 0.f;
  } else {
    if (base_score > 0) {
      // If the chosen device was idle, give a reward 1.
      score = 1.f;
    } else {
      DevID_t num_devices =
          this->device_manager_->template get_num_devices(ParlaDeviceType::CUDA);
      double total_nonidle_time{0}, current_time{0};
      double current_device_nonidle_time{0};
      // Only consider CUDA devices, and their index starts from 1.
      for (DevID_t d = 1; d < num_devices + 1; ++d) {
        ParlaDevice* device = this->device_manager_->get_device_by_global_id(d);
        auto [idle_time, nonidle_time] = device->get_total_idle_time();
        total_nonidle_time += nonidle_time;
        if (d == chosen_device_id) {
          current_time = device->current_timepoint_count_from_beginning();
          current_device_nonidle_time = nonidle_time;
        }
      }
      double ideal_time_per_device = total_nonidle_time / num_devices;
      /*
      if (current_time > 0) {
        score = ideal_time_per_device / current_time;
      } else {
        score = 0;
      }
      */

      if (current_device_nonidle_time > 0) {
        score = (ideal_time_per_device / current_device_nonidle_time);
        score = std::min(score, double{1});
      } else {
        score = 0;
      }

      std::cout << "chosen device:" << chosen_device_id <<
      " ideal:" << ideal_time_per_device << ", current time:" <<
      current_time << ", target device non-idle time:" << current_device_nonidle_time <<
      ", score:" << score << "\n";

      score = std::pow(score, 0.5);
    }

    if (score > 0 && score < 0.6) {
      score -= 1;
      std::cout << "Score became minus:" << score << "\n";
    }
    ++this->num_reward_accumulation_;
    this->reward_accumulation_ += score;
  }
  return torch::tensor({score}, torch::kDouble);
}
#endif

torch::Tensor RLEnvironment::calculate_reward_loadbalancing(
                               DevID_t chosen_device_id,
                               InnerTask* task,
                               torch::Tensor current_state) {
  double score = 0;
  size_t total_running_planned_tasks =
      this->sched_->atomic_load_total_num_mapped_tasks();
  size_t dev_running_planned_tasks =
      this->sched_->atomic_load_dev_num_mapped_tasks_device(
          chosen_device_id);
  if (dev_running_planned_tasks == 0) {
    // If the chosen device was idle, give a reward 1.
    score = 1.f;
  } else {
    if (total_running_planned_tasks != 0) {
      //score = double{1 - (dev_running_planned_tasks / double(total_running_planned_tasks))};
      score = dev_running_planned_tasks / double(total_running_planned_tasks);
      DevID_t num_devices =
          this->device_manager_->template get_num_devices(
              ParlaDeviceType::CUDA);
      // choosing device having task ratio over the total num. of tasks more than
      // (num_devices - 1) / (num_devices) is "bad" choice.
      // 0.1 is constant.
      double lower_threshold = (1 - (2 / double(num_devices)));
      double upper_threshold = (1 / double(num_devices));
      //std::cout << "threshold:" << lower_threshold << 
      //  ", upper threshold:" << upper_threshold << ", score:" << score << "\n";
      if (score >= lower_threshold) {
        score = -score;
      } else if (score <= upper_threshold) {
        // If score does not exceed the threshold, just set it to 0.
        score = (double{1} - score);
      } else {
        score = 0;
      }
    }
  }

  //std::cout << "score:" << score << ", " <<
  //  total_running_planned_tasks << ", " << dev_running_planned_tasks << "\n";

  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += score;

  return torch::tensor({score}, torch::kDouble);
}

torch::Tensor RLEnvironment::calculate_reward_parla(
                               DevID_t chosen_device_id,
                               InnerTask* task,
                               torch::Tensor current_state) {
  double score = 0;
  DevID_t num_devices =
      this->device_manager_->template get_num_devices(
          ParlaDeviceType::CUDA);
  double best_dev_score{-(std::numeric_limits<double>::max())};
  double worst_dev_score{std::numeric_limits<double>::max()};
  double chosen_dev_score{0};
  DevID_t best_dev{0};
  DevID_t worst_dev{0};
  bool found_best_dev{false};
  bool found_worst_dev{false};
#if 0
  std::cout << "Task name:" << task->name << " device selection " << chosen_device_id << "\n";
  std::cout << "best_dev :" << best_dev_score << "\n";
#endif
  for (DevID_t i = 1 ; i < num_devices + 1; ++i) {
    size_t dev_running_planned_tasks =
        this->sched_->atomic_load_dev_num_mapped_tasks_device(i);
    double device_load =
        current_state[0][DEVICE_FEATURE_OFFSET + i * NUM_DEVICE_FEATURES].item<double>(); 
    double nonlocal_data = current_state[0][
        DEVICE_FEATURE_OFFSET +
        i * NUM_DEVICE_FEATURES + 1].item<double>();
    double curr_dev_score = -(3 * device_load + nonlocal_data);
#if 0
    std::cout << "\t " << i << ", 1:" << device_load << "\n";
    std::cout << "\t " << i << ", 2:" << nonlocal_data << "\n";
#endif
    if (best_dev_score < curr_dev_score) {
      best_dev_score = curr_dev_score;
      best_dev = i;
      found_best_dev = true;
#if 0
      std::cout << "\t best score: " << best_dev_score << "\n";
      std::cout << "\t best idx: " << best_dev << "\n";
#endif
    }
    if (worst_dev_score > curr_dev_score) {
      worst_dev_score = curr_dev_score;
      worst_dev = i;
      found_worst_dev = true;
#if 0
      std::cout << "\t worst score: " << worst_dev_score << "\n";
      std::cout << "\t worst idx: " << worst_dev << "\n";
#endif
    }
    if (chosen_device_id == i) {
      chosen_dev_score = curr_dev_score;
    }
  }
  if (found_best_dev &&
          (best_dev == chosen_device_id ||
           chosen_dev_score == best_dev_score)) { score += 1; }
  if (found_worst_dev &&
          (worst_dev == chosen_device_id ||
           chosen_dev_score == worst_dev_score)) { score -= 1; }
  if (found_worst_dev && found_best_dev && worst_dev == best_dev &&
      chosen_dev_score == worst_dev_score) { score -= 1; }

  //std::cout << "Score:" << score << "\n";

  //std::cout << "score:" << score << ", " <<
  //  total_running_planned_tasks << ", " << dev_running_planned_tasks << "\n";

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
