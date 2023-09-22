#include "include/rl_environment.hpp"
#include "include/phases.hpp"
#include "include/runtime.hpp"
#include "include/profiling.hpp"

#include <chrono>
#include <cmath>

#define PRINT_LOG true 

using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

double RLEnvironment::check_task_type(InnerTask *task) {
  double task_type;
  // Expected task type is a prefix of the first '-' in a task name.
  std::string task_type_str = task->name.substr(0, task->name.find("_"));
  auto found = this->task_type_map_.find(task_type_str);
  if (found == this->task_type_map_.end()) {
    // If a task type for task does not exist, create new one.
    // The new task type is assigned (last existed task type + 1).
    task_type = this->last_task_type;
    this->task_type_map_[task_type_str] = this->last_task_type++;
  } else {
    task_type = found->second;
  }
  return task_type;
}

void RLEnvironment::make_current_workload_state(
    InnerTask *task, torch::Tensor current_state, DevID_t num_devices) {
  // 1) # of active dependencies:
  uint32_t num_considering_task_states{
      int(TaskState::RUNNING) - int(TaskState::MAPPED) + 1};
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
    TaskState task_state = task->get_state();
    if (task_state < TaskState::MAPPED ||
        task_state > TaskState::RUNNING) { continue; }
    ++num_dependencies[int(task_state) - int(TaskState::MAPPED)];

    // Number of dependencies mapped to this device:
    // This is device specific task state, not workload information.
    // To remove iterations, exceptionally fills this feature.
    for(size_t i = 0; i < dependency->assigned_devices.size(); ++i) {
      DevID_t did = dependency->assigned_devices[i]->get_global_id();
      current_state[0][
          DEVICE_FEATURE_OFFSET + did * NUM_DEVICE_FEATURES +
          NUM_DEVICE_FEATURES - 1] += 1;
    }
  }

  // 2) Total bytes of IN/INOUT/OUT PArrays.
  // XXX(hc): This assumes a single device task; so PArrays are initialized at
  // 0th device.
  double total_bytes{0};
  for (size_t pi = 0; pi < task->parray_list[0].size(); ++pi) {
    InnerPArray *parray = task->parray_list[0][pi].first;
    total_bytes += int64_t{parray->get_size()}; 
  }

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
    LOG_INFO("Debug", "Num. dependency per state {}: current state[{}]={}",
        i, i, current_state[0][i].item<double>());
  }
  LOG_INFO("Debug", "Total bytes: current state[{}]={}",
      print_offset, current_state[0][print_offset].item<double>());
  ++print_offset;
  LOG_INFO("Debug", "Task type: Current state[{}]={}",
      print_offset, current_state[0][print_offset].item<double>());
  LOG_INFO("Debug", "Current Task State Print [done]");
#endif
}

void RLEnvironment::make_current_device_specific_state(
    InnerTask *task, torch::Tensor current_state, DevID_t num_devices) {
  // 1) The number of tasks for each state (TODO(hc): for now ignore CPU).
  double nmt_m = 0, nmt_rs = 0, nmt_rd = 0, nmt_rn = 0;
  size_t num_considering_task_states{4};
  size_t offset{DEVICE_FEATURE_OFFSET + NUM_DEVICE_FEATURES};
  for (DevID_t d = 1; d < num_devices; ++d) {
    nmt_m = this->sched_->atomic_load_dev_num_tasks_mapped_states(d); 
    nmt_rs = this->sched_->atomic_load_dev_num_tasks_resreserved_states(d);
    nmt_rd = this->sched_->atomic_load_dev_num_ready_tasks(d);
    nmt_rn = this->sched_->atomic_load_dev_num_running_tasks(d);
    current_state[0][offset] = nmt_m;
    current_state[0][offset+1] = nmt_rs;
    current_state[0][offset+2] = nmt_rd;
    current_state[0][offset+3] = nmt_rn;
    offset += NUM_DEVICE_FEATURES;
  }

#if PRINT_LOG
  size_t print_offset{DEVICE_FEATURE_OFFSET + NUM_DEVICE_FEATURES};
  for (DevID_t d = 1; d < num_devices; ++d) {
    for (uint32_t st = 0; st < num_considering_task_states; ++st) {
      LOG_INFO("Debug", "Num. states {}: current state[{}]={}",
          st, print_offset + st,
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
    LOG_INFO("Debug", "Nonlocal: current state[{}]={}",
        offset - 1, current_state[0][offset - 1].item<double>());
#endif

    // 3) Reservable memory.
    ParlaDevice* device =
        this->device_manager_->get_device_by_global_id(d);
    const ResourcePool_t &device_pool = device->get_resource_pool();
    const ResourcePool_t &reserved_device_pool = device->get_reserved_pool();  
    int64_t total_device_memory = device_pool.template get<Resource::Memory>();
    int64_t remaining_device_memory =
        reserved_device_pool.template get<Resource::Memory>();
    current_state[0][offset] = (total_device_memory - remaining_device_memory) / double{1 << 20};
#if PRINT_LOG
    LOG_INFO("Debug", "Total device: current state[{}]={}, {} / {}",
        offset, current_state[0][offset].item<double>(), remaining_device_memory,
        total_device_memory);
    LOG_INFO("Debug", "Num. dependencies mapped to: current state[{}]={}",
        offset + 1, current_state[0][offset + 1].item<double>());
#endif
  }
}

torch::Tensor RLEnvironment::make_current_state(InnerTask *task) {
  DevID_t num_devices =
      this->device_manager_->template get_num_devices(ParlaDeviceType::All);
  torch::Tensor current_state =
      torch::zeros({1, NUM_TASK_FEATURES +
          (num_devices * NUM_DEVICE_FEATURES)}, torch::kDouble);
#if PRINT_LOG
  LOG_INFO("Debug", "current state {}", task->name);
#endif
  this->make_current_workload_state(task, current_state, num_devices);
  this->make_current_device_specific_state(task, current_state, num_devices);

  return current_state;
}

torch::Tensor RLEnvironment::make_next_state(
    torch::Tensor current_state, DevID_t chosen_device_id, InnerTask *task) {
  torch::Tensor next_state = current_state.clone();
  ParlaDevice* device =
      this->device_manager_->get_device_by_global_id(chosen_device_id);
  size_t offset = DEVICE_FEATURE_OFFSET + chosen_device_id * NUM_DEVICE_FEATURES;
  // The first element of a device is the number of task mapped and
  // running on that device. 
  next_state[0][offset] += 1;
  // All PArrays would be materialized.
  next_state[0][offset + 4] = 0;
  // Reservable memory.
  const ResourcePool_t &device_pool = device->get_resource_pool();
  ResourcePool_t &task_pool = task->device_constraints[chosen_device_id];
  const ResourcePool_t &reserved_device_pool = device->get_reserved_pool();  
  int64_t total_device_memory = device_pool.template get<Resource::Memory>();
  // Reserved device memory count is not updated yet.
  // After all of the RL phases are done, this count is updated based on
  // task memory.
  double prev_nonlocal_bytes{0};
  for (size_t pi = 0; pi < task->parray_list[0].size(); ++pi) {
    InnerPArray *parray = task->parray_list[0][pi].first;
    if (!this->parray_tracker_->
            get_parray_state(chosen_device_id, parray->parent_id)) {
      prev_nonlocal_bytes += double{parray->get_size()};
    }
  }
  // Nonlocal PArrays are moved to the target device.
  int64_t remaining_device_memory =
      reserved_device_pool.template get<Resource::Memory>() - prev_nonlocal_bytes;
  next_state[0][offset + 5] =
      (total_device_memory - remaining_device_memory) / double(1 << 20);

#if PRINT_LOG
  LOG_INFO("Debug", "next state {}", task->name);
  LOG_INFO("Debug", "[# mapped tasks] next state[{}]={}",
      offset, next_state[0][offset].item<double>());
  LOG_INFO("Debug", "[nonlocal bytes] next state[{}]={}",
      offset + 4, next_state[0][offset + 4].item<double>());
  LOG_INFO("Debug", "[device remaining bytes] next state[{}]={}, {} / {}",
      offset + 5, next_state[0][offset + 5].item<double>(),
      remaining_device_memory, total_device_memory);
#endif

  return next_state;
}

torch::Tensor RLEnvironment::calculate_reward(DevID_t chosen_device_id,
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
  double total_num_mapped_tasks =
      this->sched_->atomic_load_total_num_mapped_tasks();
  size_t num_considering_task_states{4};
#if PRINT_LOG
  LOG_INFO("Debug", "Task {} score calculation", task->name);
#endif
  for (DevID_t i = 1 ; i < num_devices + 1; ++i) {
    double dev_running_planned_tasks =
        this->sched_->atomic_load_dev_num_mapped_tasks_device(i);
    double norm_dev_load = dev_running_planned_tasks / total_num_mapped_tasks;
    double norm_nonlocal_data = current_state[0][
        DEVICE_FEATURE_OFFSET + i * NUM_DEVICE_FEATURES +
        num_considering_task_states].item<double>();
    double curr_dev_score = -(30 * norm_dev_load + 10 * norm_nonlocal_data);
    if (best_dev_score < curr_dev_score) {
      best_dev_score = curr_dev_score;
      best_dev = i;
      found_best_dev = true;
    }
    if (worst_dev_score > curr_dev_score) {
      worst_dev_score = curr_dev_score;
      worst_dev = i;
      found_worst_dev = true;
    }
    if (chosen_device_id == i) {
      chosen_dev_score = curr_dev_score;
    }
#if PRINT_LOG
    LOG_INFO("Debug",
        "Devie ID {} norm. dev. load: {}, norm. nonlocal data:{}, score: {}"
        ,i, norm_dev_load, norm_nonlocal_data, curr_dev_score);
#endif
  }
  if (found_best_dev &&
          (best_dev == chosen_device_id ||
           chosen_dev_score == best_dev_score)) { score += 1; }
  if (found_worst_dev &&
          (worst_dev == chosen_device_id ||
           chosen_dev_score == worst_dev_score)) { score -= 1; }
  if (found_worst_dev && found_best_dev && worst_dev == best_dev &&
      chosen_dev_score == worst_dev_score) { score -= 1; }

#if PRINT_LOG
    LOG_INFO("Debug", "last score:{}", score);
#endif

  ++this->num_reward_accumulation_;
  this->reward_accumulation_ += score;

  return torch::tensor({score}, torch::kDouble);
}

void RLEnvironment::output_reward(size_t episode) {
  std::cout << "Accumulated reward:" << this->reward_accumulation_ << ", and #:" << this->num_reward_accumulation_ << "\n";
  std::ofstream fp("reward.out", std::ios_base::app);
  fp << episode << ", " << (this->reward_accumulation_ / this->num_reward_accumulation_) << "\n";
  this->reward_accumulation_ = 0;
  this->num_reward_accumulation_ = 0;
  fp.close();
}
