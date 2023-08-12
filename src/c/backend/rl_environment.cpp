#include "include/rl_environment.hpp"
#include "include/phases.hpp"
#include "include/runtime.hpp"

#include <chrono>

#define NUM_TASK_FEATURES 4
#define NUM_DEP_TASK_FEATURES 4
#define NUM_DEVICE_FEATURES 9
#define DEVICE_FEATURE_OFFSET (NUM_TASK_FEATURES + NUM_DEP_TASK_FEATURES)

using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

double RLEnvironment::check_task_type(InnerTask *task) {
  double task_type;
  std::string task_type_str = task->name.substr(0, task->name.find("-"));
  auto found = this->task_type_map.find(task_type_str);
  if (found == this->task_type_map.end()) {
    task_type = this->last_task_type;
    std::cout << task_type_str << "'s task type found \n";
    this->task_type_map[task_type_str] = this->last_task_type++;
  } else {
    task_type = found->second;
  }
  std::cout << task_type_str << "'s task type:" << task_type << "\n";
  return task_type;
}

void RLEnvironment::make_current_task_state(
    InnerTask *task, torch::Tensor current_state, DevID_t num_devices,
    size_t offset, bool accum) {
  // 1) # of active dependencies:
  int64_t num_active_dependencies{0};
  for (size_t i = 0; i < task->dependencies.size(); ++i) {
    // Access each dependency of the compute task.
    InnerTask *dependency = task->dependencies.at(i);
    if (dependency->is_data.load()) { continue; }
    if (dependency->get_state() < Task::State::COMPLETED) {
      ++num_active_dependencies;
    }
  }

  // 2) # of visible 1-hop dependents.
  int64_t num_active_dependents{0};
  for (size_t i = 0; i < task->dependents.size(); ++i) {
    // Dependent tasks do not have data movement tasks yet.
    // (Those will be created after this task creates them first)
    InnerTask *dependent = task->dependents.at(i);
    if (dependent->name.find("global_0") != std::string::npos) { continue; }
    if (dependent->get_state() < Task::State::COMPLETED) {
      ++num_active_dependents;
    }
  }

  // 3) Mega-Bytes of IN/INOUT/OUT data.
  double total_bytes{0};
  for (DevID_t d = 0; d < num_devices; ++d) {
    for (size_t pi = 0; pi < task->parray_list[d].size(); ++pi) {
      InnerPArray *parray = task->parray_list[d][pi].first;
      total_bytes += int64_t{parray->get_size()}; 
    }
  }

  if (accum) {
    current_state[0][offset] += num_active_dependencies;
    current_state[0][offset + 1] += num_active_dependents;
    current_state[0][offset + 2] += (total_bytes / double{1 << 20});
  } else {
    current_state[0][offset] = num_active_dependencies;
    current_state[0][offset + 1] = num_active_dependents;
    current_state[0][offset + 2] = (total_bytes / double{1 << 20});
    // 4) Task type.
    double task_type = this->check_task_type(task);
    current_state[0][offset + 3] = task_type;
  }
}

void RLEnvironment::make_current_device_state(
    InnerTask *task, torch::Tensor current_state, DevID_t num_devices) {
  double total_bytes{0}, local_bytes{0};
  // 0) Total IN/INOUT/OUT data bytes
  for (size_t pi = 0; pi < task->parray_list[0].size(); ++pi) {
    InnerPArray *parray = task->parray_list[0][pi].first;
    total_bytes += double{parray->get_size()};
  }

  double total_idle_time{0}, total_nonidle_time{0};
	TimePoint current_time_point = std::chrono::system_clock::now();
  for (DevID_t d = 1; d < num_devices; ++d) {
    ParlaDevice* device = this->device_manager_->get_device_by_global_id(d);
    auto [idle_time, nonidle_time] = device->get_total_idle_time(current_time_point);
    total_idle_time += idle_time;
    total_nonidle_time += nonidle_time;
  }

  // Exclude CPU
  for (DevID_t d = 1; d < num_devices; ++d) {
    ParlaDevice* device = this->device_manager_->get_device_by_global_id(d);
    double nonlocal_bytes{0};
    // 1) # of tasks on queues.
    int64_t dev_running_planned_tasks =
        this->mapper_->atomic_load_dev_num_mapped_tasks_device(d);
    current_state[0][DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES] =
        dev_running_planned_tasks;
    if (dev_running_planned_tasks > 0) {
      std::cout << "Device " << d << " remote bytes:" <<
        device->get_remote_data_bytes() << ", num dependencies:" << device->get_num_dependencies() <<
        ", num dependents:" << device->get_num_dependents() << "\n";
      current_state[0][DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES + 5] = device->get_remote_data_bytes() / dev_running_planned_tasks;
      current_state[0][DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES + 6] = device->get_num_dependencies() / dev_running_planned_tasks;
      current_state[0][DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES + 7] = device->get_num_dependents() / dev_running_planned_tasks;
    }
    // 2) Data locality.
    if (total_bytes > 0) {
      for (size_t pi = 0; pi < task->parray_list[0].size(); ++pi) {
        InnerPArray *parray = task->parray_list[0][pi].first;
        if (!this->parray_tracker_->get_parray_state(d, parray->parent_id)) {
          nonlocal_bytes += double{parray->get_size()};  
        }
      }
      current_state[0][DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES + 1] =
          nonlocal_bytes / total_bytes;
    } else {
      current_state[0][DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES + 1] = 0;
    }

    // 3) Reservable memory.
    const ResourcePool_t &device_pool = device->get_resource_pool();
    const ResourcePool_t &reserved_device_pool = device->get_reserved_pool();  
    int64_t total_memory = device_pool.get(Resource::Memory);
    int64_t remaining_memory = reserved_device_pool.get(Resource::Memory);
    current_state[0][DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES + 2] =
      (total_memory > 0)? (remaining_memory / double{total_memory}) : 0;

    // 4) Relative idle/non-idle time.
    auto [idle_time, nonidle_time] = device->get_total_idle_time(current_time_point);
    if (total_idle_time > 0) {
      current_state[0][DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES + 3] = idle_time / total_idle_time;
      std::cout << "device " << d << "'s idle time:" << idle_time << ", " << idle_time / total_idle_time << "\n";
    }
    
    if (total_nonidle_time > 0) {
      current_state[0][DEVICE_FEATURE_OFFSET + d * NUM_DEVICE_FEATURES + 4] = nonidle_time/ total_nonidle_time;
      std::cout << "device " << d << "'s nonidle time:" << nonidle_time << ", " << nonidle_time / total_nonidle_time << "\n";
    }
  }
}

void RLEnvironment::make_current_active_deptask_state(
    InnerTask *task, torch::Tensor current_state, DevID_t num_devices) {
  int64_t num_active_dependencies{0};
  std::unordered_map<Task::State, size_t> state_counter;
  size_t major_state_counter{std::numeric_limits<size_t>::min()};
  Task::State major_state{Task::State::CREATED};
  double num_active_tasks{0};
  // Iter 0: Dependencies
  // Iter 1: Dependents
  for (size_t t = 0; t < 2; ++t) {
    auto& dep_task_list = (t == 0)? task->dependencies : task->dependents;
    for (size_t i = 0; i < dep_task_list.size(); ++i) {
      // Access each dependency/dependent of the compute task.
      InnerTask *dep_task = dep_task_list.at(i);
      if (dep_task->is_data.load()) { continue; }
      if (dep_task->get_state() < Task::State::COMPLETED) {
        ++num_active_tasks;
        this->make_current_task_state(
            task, current_state, num_devices, NUM_TASK_FEATURES, true);
        auto found = state_counter.find(task->get_state());
        if (found == state_counter.end()) {
          state_counter[task->get_state()] = 0;
        } else {
          size_t& sc = state_counter[task->get_state()];
          ++sc;
          if (major_state_counter < sc) {
            major_state_counter = sc;
            major_state = task->get_state();
          }
        }
      }
    }
  }
  current_state[0][NUM_TASK_FEATURES + NUM_DEP_TASK_FEATURES - 1] = major_state;
  for (size_t i = 0; i < NUM_TASK_FEATURES; ++i) {
    double old_value = current_state[0][NUM_TASK_FEATURES + i].item<double>();
    if (num_active_tasks > 0) {
      current_state[0][NUM_TASK_FEATURES + i] = old_value / double{num_active_tasks};
    }
  }
}

torch::Tensor RLEnvironment::make_current_state(InnerTask *task) {
  DevID_t num_devices =
      this->device_manager_->template get_num_devices(ParlaDeviceType::All);
  torch::Tensor current_state =
      torch::zeros({1,
          NUM_TASK_FEATURES + NUM_DEP_TASK_FEATURES +
          (num_devices * NUM_DEVICE_FEATURES)}, torch::kDouble);
  this->make_current_task_state(task, current_state, num_devices, 0, false);
  this->make_current_active_deptask_state(task, current_state, num_devices);
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
      reserved_device_pool.get(Resource::Memory) + task_memory;
  next_state[0][
      DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES + 2] =
      (total_memory > 0)? (remaining_memory / double{total_memory}) : 0;

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

#if 0
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
