#ifndef PARLA_RL_TASK_MAPPER
#define PARLA_RL_TASK_MAPPER

#include "device_manager.hpp"
#include "rl_environment.hpp"
#include "runtime.hpp"
#include "policy.hpp"

#include <random>
#include <torch/torch.h>

class Mapper;

class ExperienceReplay {
public:
  using BufferTupleType =
      std::tuple<torch::Tensor /* Current state */,
                 torch::Tensor /* Chosen device from the state */,
                 torch::Tensor /* Next state */, torch::Tensor /* Reward */,
                 uint64_t /* Episode */>;
  using BufferTy = std::deque<BufferTupleType>;
  ExperienceReplay(int64_t capacity) : capacity_(capacity) {}

  /*
   * Append new tuple to the experience replay buffer.
   *
   * @param curr_state Tensor of the current state
   * @param chosen_device Tensor of the chosen device
   * @param next_state Tensor of the next state
   * @param reward Tensor of the reward from the chosen device
   * @param episode Current episode
   */
  void push(torch::Tensor curr_state, torch::Tensor chosen_device,
            torch::Tensor next_state, torch::Tensor reward, uint64_t episode) {
    BufferTupleType new_buffer_element =
        std::make_tuple(curr_state, chosen_device, next_state, reward, episode);
    while (this->buffer_.size() >= this->capacity_) {
      this->buffer_.pop_front();
    }
    this->buffer_.push_back(new_buffer_element);
  }

  std::vector<BufferTupleType> sample(int64_t batch_size) {
    std::vector<BufferTupleType> sampled_buffer(batch_size);
    std::sample(this->buffer_.begin(), this->buffer_.end(),
                sampled_buffer.begin(), sampled_buffer.size(),
                std::mt19937_64{std::random_device{}()});
    return sampled_buffer;
  }

  void print() {
    for (size_t i = 0; i < this->buffer_.size(); ++i) {
      std::cout << "\n [" << i << "th buffer]\n";
      std::cout << "\t Current state: " << std::get<0>(this->buffer_[i]) << ", "
                << "Chosen device: " << std::get<1>(this->buffer_[i]) << ", "
                << "Next state: " << std::get<2>(this->buffer_[i]) << ", "
                << "Reward: " << std::get<3>(this->buffer_[i]) << "\n";
    }
  }

  size_t size() { return this->buffer_.size(); }

private:
  int64_t capacity_;
  BufferTy buffer_;
};

struct FullyConnectedDQN : torch::nn::Module {
  FullyConnectedDQN(size_t in_dim, size_t out_dim)
      : in_dim_(in_dim), out_dim_(out_dim), device_(torch::kCUDA) {
    fc1_ = register_module("fc1", torch::nn::Linear(in_dim, in_dim * 4));
    fc1_->to(device_);
    fc2_ = register_module("fc2", torch::nn::Linear(in_dim * 4, in_dim * 4));
    fc2_->to(device_);
    out_ = register_module("out", torch::nn::Linear(in_dim * 4, out_dim));
    out_->to(device_);
  }

  torch::Tensor forward(torch::Tensor x) {
    // std::cout << "forward 1:" << x << "\n";
    x = x.to(device_);
    // std::cout << "forward 2:" << x << "\n";
    // std::cout << "in dim:" << in_dim_ << "\n";
    // std::cout << "reshaped:" << x.reshape({x.size(0), in_dim_}) << "\n";
    x = torch::relu(fc1_->forward(x.reshape({x.size(0), in_dim_})));
    // std::cout << "state 1:" << x << "\n";
    x = torch::relu(fc2_->forward(x.reshape({x.size(0), in_dim_ * 4})));
    // std::cout << "state 2:" << x << "\n";
    x = out_->forward(x.reshape({x.size(0), in_dim_ * 4}));
    std::cout << "out:" << x << "\n";
    // std::cout << "out after squeeze:" << torch::squeeze(x, 1) << "\n";
    return x.squeeze(0);
  }

  uint32_t in_dim_, out_dim_;
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, out_{nullptr};
  torch::Device device_;
};

class RLAgent {
public:
  using BufferTupleType = typename ExperienceReplay::BufferTupleType;

  RLAgent(size_t in_dim, size_t out_dim, uint32_t n_actions,
          torch::Device device = torch::kCUDA,
          std::string rl_mode = "training",
          float eps_start = 0.9, float eps_end = 0.05, float eps_decay = 200,
          size_t batch_size = 2, float gamma = 0.999)
      : policy_net_(in_dim, out_dim), target_net_(in_dim, out_dim),
        device_(device), rl_mode_(rl_mode), n_actions_(n_actions),
        eps_start_(eps_start), eps_end_(eps_end), eps_decay_(eps_decay),
        batch_size_(batch_size), gamma_(gamma), steps_(0),
        replay_memory_(1000),
        rms_optimizer(policy_net_.parameters(), torch::optim::RMSpropOptions(0.025)) {}

  std::pair<uint32_t, bool> select_device(torch::Tensor state,
                         std::vector<ParlaDevice *> device_candidates,
                         std::vector<bool> *mask = nullptr) {
    float eps_threshold =
        this->eps_end_ + (this->eps_start_ - this->eps_end_) *
                             exp(-1.f * this->steps_ / this->eps_decay_);
    this->steps_ += 1;
    // Random number generation.
    std::uniform_real_distribution<double> distribution(0, 1);
    std::mt19937_64 mt(random());
    float sample = distribution(mt);
    std::cout << "Select device from policy " << eps_threshold
              << " sample: " << sample << " \n";
    std::cout << "state original:" << state << "\n";
    // TODO(hc): remove it
    eps_threshold = 0;
    //if (sample > eps_threshold) {
      torch::NoGradGuard no_grad;
      torch::Tensor out_tensor = this->policy_net_.forward(state);
      std::cout << "out tensor:" << out_tensor << "\n";
      std::cout << " n actions:" << this->n_actions_ << "\n";
      for (uint32_t action = 0; action < this->n_actions_; ++action) {
        auto max_action_pair = out_tensor.max(0);
        torch::Tensor max_tensor = std::get<0>(max_action_pair);
        int64_t max_tensor_idx = (std::get<1>(max_action_pair)).item<int64_t>();
        std::cout << "max:" << std::get<0>(max_action_pair) <<
            ", index:" << max_tensor_idx << "\n";

        // If mask is null, it means there is no constraint in device selection.
        // If mask is not null, this task has device candidates and should
        // follw that.
        if (mask == nullptr || (mask != nullptr && (*mask)[max_tensor_idx])) {
          return std::make_pair(max_tensor_idx, true);
        } else {
          out_tensor[max_tensor_idx] = -999999;
        }
      }
      /*
    } else {
      // TODO: replace it with candidates.
      std::uniform_real_distribution<> randomly_chosen_device(0.f, 5.f);
      return std::make_pair(
          uint32_t{randomly_chosen_device(mt)}, std::numeric_limits<float>::max());
    }
    */
    return std::make_pair(0, false);
  }

  void optimize_model(uint64_t rl_episodes) {
    if (this->replay_memory_.size() < this->batch_size_) {
      return;
    }

    std::cout << "Model optimization starts\n";

    std::vector<BufferTupleType> batch =
        this->replay_memory_.sample(this->batch_size_);
    std::vector<torch::Tensor> curr_states;
    std::vector<torch::Tensor> next_states;
    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> rewards;

    for (BufferTupleType &buffer : batch) {
      curr_states.push_back(std::get<0>(buffer));
      next_states.push_back(std::get<2>(buffer));
      actions.push_back(std::get<1>(buffer));
      rewards.push_back(std::get<3>(buffer));
    }

    torch::Tensor curr_states_tensor;
    torch::Tensor next_states_tensor;
    torch::Tensor actions_tensor;
    torch::Tensor rewards_tensor;

    curr_states_tensor = torch::cat(curr_states, 0);
    next_states_tensor = torch::cat(next_states, 0);
    actions_tensor = torch::cat(actions, 0).to(this->device_).unsqueeze(0);
    rewards_tensor = torch::cat(rewards, 0).to(this->device_);

    // Calculate expected Q value.
    torch::Tensor q_values = this->policy_net_.forward(curr_states_tensor);
    q_values = q_values.gather(1, actions_tensor).reshape({this->batch_size_, 1});
    torch::Tensor next_target_q_values =
        this->target_net_.forward(next_states_tensor);
    next_target_q_values = std::get<0>(next_target_q_values.max(1));
    torch::Tensor expected_q_values =
        this->gamma_ * next_target_q_values + rewards_tensor;
    std::cout << " qvals:" << q_values << ", "
              << " action_tensor:" << actions_tensor << ", "
              << " action_tensor unsqueezed:" << actions_tensor.unsqueeze(1)
              << ", reward tensor:" << rewards_tensor << ", "
              << ", expected q values:" << expected_q_values << "\n";

    torch::Tensor loss = torch::smooth_l1_loss(q_values, expected_q_values.reshape({this->batch_size_, 1}));
    // Zerofying gradients in the optimizer.
    this->rms_optimizer.zero_grad();
    // Update gradients.
    loss.backward();
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      parameter.grad().data().clamp(-1, 1);
    }
    this->rms_optimizer.step();
  }

  void append_replay_memory(torch::Tensor curr_state,
                            torch::Tensor chosen_device,
                            torch::Tensor next_state, torch::Tensor reward,
                            uint64_t episode) {
    this->replay_memory_.push(curr_state, chosen_device, next_state, reward,
                              episode);
  }

  void print() { this->replay_memory_.print(); }

  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, uint64_t>>
  sample(int64_t b) {
    return this->replay_memory_.sample(b);
  }

private:
  // TODO: replay memory

  FullyConnectedDQN policy_net_;
  FullyConnectedDQN target_net_;
  torch::Device device_;
  std::string rl_mode_;
  uint32_t n_actions_;
  float eps_start_, eps_end_, eps_decay_;
  size_t batch_size_;
  float gamma_;
  uint64_t steps_;
  ExperienceReplay replay_memory_;
  torch::optim::RMSprop rms_optimizer;
};

class RLTaskMappingPolicy : public MappingPolicy {
public:
  RLTaskMappingPolicy(
      DeviceManager *device_manager, PArrayTracker *parray_tracker,
      Mapper *mapper);

  bool calc_score_devplacement(
      InnerTask *task,
      const std::shared_ptr<DeviceRequirement> &dev_placement_req,
      const Mapper &mapper, Score_t *score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
          &parray_list) override;

  bool calc_score_archplacement(
      InnerTask *task, ArchitectureRequirement *arch_placement_req,
      const Mapper &mapper, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
      Score_t *chosen_dev_score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
          &parray_list,
      std::vector<bool> *is_dev_assigned = nullptr) override;

  bool calc_score_mdevplacement(
      InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
      const Mapper &mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
      Score_t *average_score,
      const std::vector<
          std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list) override;

  void run_task_mapping(
      InnerTask *task, const Mapper &mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *chosen_devices,
      const std::vector<std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list,
      std::vector<std::shared_ptr<PlacementRequirementBase>>
          *placement_req_options_vec) override;
#if 0
  // RL forwarding.
  bool LocalityLoadBalancingMappingPolicy::calc_score_devplacement();
  // RL forwarding.
  // Sets devices of different architecture to max value.
  // (so the load is infinite, to ignore that)
  bool LocalityLoadBalancingMappingPolicy::calc_score_archplacement();
  // Find max device first, 
  // Then set that device state to max value.
  // Repeat until it finds all device combinations
  // State should be passed to here.
  bool LocalityLoadBalancingMappingPolicy::calc_score_mdevplacement();
#endif
private:
  /// RL agent.
  RLAgent *rl_agent_;
  /// RL environment.
  RLEnvironment *rl_env_;
  torch::Tensor rl_current_state_;
  torch::Tensor rl_next_state_;
};

#endif
