#ifndef PARLA_RL_TASK_MAPPER
#define PARLA_RL_TASK_MAPPER

#include "device_manager.hpp"

#include <torch/torch.h>
#include <random>

class ExperienceReplay {
public:
  using BufferTupleType = std::tuple<torch::Tensor /* Current state */,
                                     torch::Tensor /* Chosen device from the state */,
                                     torch::Tensor /* Next state */,
                                     torch::Tensor /* Reward */,
                                     uint64_t      /* Episode */>;
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
            torch::Tensor next_state, torch::Tensor reward,
            uint64_t episode) {
    BufferTupleType new_buffer_element = std::make_tuple(
        curr_state, chosen_device, next_state, reward, episode);
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
      std::cout << "\t Current state: " << std::get<0>(this->buffer_[i]) << ", " <<
                   "Chosen device: " << std::get<1>(this->buffer_[i]) << ", " <<
                   "Next state: " << std::get<2>(this->buffer_[i]) << ", " <<
                   "Reward: " << std::get<3>(this->buffer_[i]) << "\n";
    }
  }
            
private:
  int64_t capacity_;
  BufferTy buffer_;
};

struct FullyConnectedDQN : torch::nn::Module {
  FullyConnectedDQN(size_t in_dim, size_t out_dim) :
      in_dim_(in_dim), out_dim_(out_dim), device_(torch::kCUDA) {
    fc1_ = register_module("fc1", torch::nn::Linear(in_dim, in_dim * 4));
    fc1_->to(device_);
    fc2_ = register_module("fc2", torch::nn::Linear(in_dim * 4, in_dim * 4));
    fc2_->to(device_);
    out_ = register_module("out", torch::nn::Linear(in_dim * 4, out_dim));
    out_->to(device_);
  }

  torch::Tensor forward(torch::Tensor x) {
    //std::cout << "forward 1:" << x << "\n";
    x = x.to(device_);
    //std::cout << "forward 2:" << x << "\n";
    //std::cout << "in dim:" << in_dim_ << "\n";
    //std::cout << "reshaped:" << x.reshape({x.size(0), in_dim_}) << "\n";
    x = torch::relu(fc1_->forward(x.reshape({x.size(0), in_dim_})));
    //std::cout << "state 1:" << x << "\n";
    x = torch::relu(fc2_->forward(x.reshape({x.size(0), in_dim_ * 4})));
    //std::cout << "state 2:" << x << "\n";
    x = out_->forward(x.reshape({x.size(0), in_dim_ * 4}));
    std::cout << "out:" << x << "\n";
    //std::cout << "out after squeeze:" << torch::squeeze(x, 1) << "\n";
    return x.squeeze(0);
  }

  uint32_t in_dim_, out_dim_;
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, out_{nullptr};
  torch::Device device_;
};

class RLAgent {
public:
  RLAgent(size_t in_dim, size_t out_dim,
        torch::Device device = torch::kCUDA,
        std::string rl_mode = "training", uint32_t n_actions = 4,
        float eps_start = 0.9, float eps_end = 0.05,
        float eps_decay = 200) :
      policy_net_(in_dim, out_dim), target_net_(in_dim, out_dim),
      device_(device), rl_mode_(rl_mode), n_actions_(n_actions),
      eps_start_(eps_start), eps_end_(eps_end), eps_decay_(eps_decay),
      steps_(0), replay_memory_(1000) {}

  uint32_t select_device(
      torch::Tensor state, std::vector<ParlaDevice *> device_candidates) {
    float eps_threshold = this->eps_end_ + (this->eps_start_ - this->eps_end_) *
                              exp(-1.f * this->steps_ / this->eps_decay_);
    this->steps_ += 1;
    // Random number generation.
    std::uniform_real_distribution<double> distribution(0, 1);
    std::mt19937_64 mt(random());
    float sample = distribution(mt);
    std::cout << "Select device from policy " << eps_threshold << 
      " sample: " << sample << " \n";
    std::cout << "state original:" << state << "\n";
    // TODO(hc): remove it
    eps_threshold = 0;
    if (sample > eps_threshold) {
      torch::NoGradGuard no_grad;
      torch::Tensor out_tensor = this->policy_net_.forward(state);
      //std::cout << "out tensor:" << out_tensor << "\n";
      for (uint32_t action = 0; action < this->n_actions_; ++action) {
        auto max_action_pair = out_tensor.max(0);
        torch::Tensor max_tensor = std::get<0>(max_action_pair);
        int64_t max_tensor_idx = (std::get<1>(max_action_pair)).item<int64_t>();
        //std::cout << "max:" << std::get<0>(max_action_pair) <<
        //    ", index:" << max_tensor_idx << "\n";

        // TODO: replace it with candidates.
        if (max_tensor_idx < 4) {
          return max_tensor_idx;
        } else {
          out_tensor[max_tensor_idx] = -999999;
        }
        //std::cout << "updated tensor:" << out_tensor << "\n";
      }
    } else {
      // TODO: replace it with candidates.
      std::uniform_real_distribution<> randomly_chosen_device(0.f, 5.f);
      return uint32_t{randomly_chosen_device(mt)};
    }
  }

  void optimize_model(uint64_t rl_episodes) {
    // TODO: 

  }

  void append_replay_memory(
      torch::Tensor curr_state, torch::Tensor chosen_device,
      torch::Tensor next_state, torch::Tensor reward,
      uint64_t episode) {
    this->replay_memory_.push(curr_state, chosen_device, next_state, reward, episode);
  }

  void print() {
    this->replay_memory_.print();
  }

  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    uint64_t>> sample(int64_t b) {
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
  uint64_t steps_;
  ExperienceReplay replay_memory_;
};

#endif
