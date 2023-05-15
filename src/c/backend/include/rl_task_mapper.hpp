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

  /*
   * Sample tuples (S, S', R, A) from the experience replay buffer.
   *
   * @param batch_size Batch size
   */
  std::vector<BufferTupleType> sample(int64_t batch_size) {
    std::vector<BufferTupleType> sampled_buffer(batch_size);
    std::sample(this->buffer_.begin(), this->buffer_.end(),
                sampled_buffer.begin(), sampled_buffer.size(),
                std::mt19937_64{std::random_device{}()});
    return sampled_buffer;
  }

  /*
   * Return the number of tuples in the experience replay memory.
   */
  size_t size() { return this->buffer_.size(); }

  void clear() {
    this->buffer_.clear();
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

private:
  /// The maximum experience replay memory size.
  int64_t capacity_;
  /// Experience replay memory.
  BufferTy buffer_;
};

struct FullyConnectedDQNImpl : public torch::nn::Module {
  FullyConnectedDQNImpl(size_t in_dim, size_t out_dim)
      : in_dim_(in_dim), out_dim_(out_dim), device_(torch::kCUDA) {
    // Move networks to a GPU.
    fc1_ = register_module("fc1", torch::nn::Linear(in_dim, in_dim * 4));
    fc1_->to(device_);
    fc2_ = register_module("fc2", torch::nn::Linear(in_dim * 4, in_dim * 4));
    fc2_->to(device_);
    out_ = register_module("out", torch::nn::Linear(in_dim * 4, out_dim));
    out_->to(device_);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.to(device_);
    x = torch::relu(fc1_->forward(x.reshape({x.size(0), in_dim_})));
    x = torch::relu(fc2_->forward(x.reshape({x.size(0), in_dim_ * 4})));
    x = out_->forward(x.reshape({x.size(0), in_dim_ * 4}));
    return x.squeeze(0);
  }

  uint32_t in_dim_, out_dim_;
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, out_{nullptr};
  torch::Device device_;
};

//TORCH_MODULE(FullyConnectedDQN);

class RLAgent {
public:
  using BufferTupleType = typename ExperienceReplay::BufferTupleType;


  RLAgent(size_t in_dim, size_t out_dim, uint32_t n_actions,
          torch::Device device = torch::kCUDA,
          std::string rl_mode = "training",
          float eps_start = 0.9, float eps_end = 0.05, float eps_decay = 200,
          size_t batch_size = 100, float gamma = 0.999)
      : policy_net_(in_dim, out_dim), target_net_(in_dim, out_dim),
        device_(device), rl_mode_(rl_mode), n_actions_(n_actions),
        eps_start_(eps_start), eps_end_(eps_end), eps_decay_(eps_decay),
        batch_size_(batch_size), gamma_(gamma), steps_(0),
        replay_memory_(1000),
        rms_optimizer_(policy_net_.parameters(), torch::optim::RMSpropOptions(0.025)),
        episode_(0) {
    this->load_models();      
  }

  ~RLAgent() {
    this->save_models();
  }

  void save_models() {
    std::cout << "Save models..\n";
    torch::serialize::OutputArchive policy_net_output_archive;
    torch::serialize::OutputArchive target_net_output_archive;
    this->policy_net_.save(policy_net_output_archive);
    this->target_net_.save(target_net_output_archive);
    policy_net_output_archive.save_to("policy_net.pt");
    target_net_output_archive.save_to("target_net.pt");
    torch::save(this->rms_optimizer_, "rms_optimizer.pt");
#if 0
    std::ofstream fp_p("policy_net.out");
    size_t p_i{0};
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      fp_p << p_i++ << ":" << parameter << "\n";
    }
    fp_p.close();

    std::ofstream fp_t("target_net.out");
    p_i = 0;
    for (torch::Tensor parameter : this->target_net_.parameters()) {
      fp_t << p_i++ << ":" << parameter << "\n";
    }
    fp_t.close();

    std::ofstream fp_o("optimizer.out");
    p_i = 0;
    for (torch::Tensor parameter : this->rms_optimizer_.parameters()) {
      fp_o << p_i++ << ":" << parameter << "\n";
    }
    fp_o.close();
#endif
    //torch::save(this->policy_net_, "policy_net.pt");
    //torch::save(this->target_net_, "target_net.pt");
  }

  void load_models() {
    std::cout << "Load models..\n";
    if (std::ifstream fp("policy_net.pt"); fp) {
      std::cout << "Load policy net\n";
      torch::serialize::InputArchive input_archive;
      input_archive.load_from("policy_net.pt");
      this->policy_net_.load(input_archive);
      //torch::load(this->policy_net_, "policy_net.pt");
    }
    if (std::ifstream fp("target_net.pt"); fp) {
      std::cout << "Load target net\n";
      torch::serialize::InputArchive input_archive;
      input_archive.load_from("target_net.pt");
      this->target_net_.load(input_archive);
      //torch::load(this->target_net_, "target_net.pt");
    }
#if 0
    if (std::ifstream fp("rms_optimizer.pt"); fp) {
      std::cout << "Load RMS optimizer\n";
      torch::load(this->rms_optimizer_, "rms_optimizer.pt");
    }
    std::ofstream fp_p("policy_net.in");
    size_t p_i{0};
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      fp_p << p_i++ << ":" << parameter << "\n";
    }
    fp_p.close();

    std::ofstream fp_t("target_net.in");
    p_i = 0;
    for (torch::Tensor parameter : this->target_net_.parameters()) {
      fp_t << p_i++ << ":" << parameter << "\n";
    }
    fp_t.close();

    std::ofstream fp_o("optimizer.in");
    p_i = 0;
    for (torch::Tensor parameter : this->rms_optimizer_.parameters()) {
      fp_o << p_i++ << ":" << parameter << "\n";
    }
    fp_o.close();
#endif
  }

  /*
   * Select a device from the current state.
   */
  DevID_t select_device(torch::Tensor state,
                        std::vector<ParlaDevice *> device_candidates,
                        std::vector<bool> *mask = nullptr) {
    // Random number generation.
    std::uniform_real_distribution<double> distribution(0, 1);
    std::mt19937_64 mt(random());
    float sample = distribution(mt);
    float eps_threshold =
        this->eps_end_ + (this->eps_start_ - this->eps_end_) *
                             exp(-1.f * this->steps_ / this->eps_decay_);
    this->steps_ += 1;
    if (sample > eps_threshold) {
      torch::NoGradGuard no_grad;
      torch::Tensor out_tensor = this->policy_net_.forward(state);
      std::cout << ">> Out:" << out_tensor.unsqueeze(0) << "\n";
      int64_t max_tensor_idx{1};
      for (size_t a = 0; a < this->n_actions_; ++a) {
        auto max_action_pair = out_tensor.max(0);
        torch::Tensor max_tensor = std::get<0>(max_action_pair);
        max_tensor_idx = (std::get<1>(max_action_pair)).item<int64_t>();
        if (mask == nullptr || (mask != nullptr && (*mask)[max_tensor_idx])) {
          //std::cout << "\t" << max_tensor_idx << "\n";
          return static_cast<DevID_t>(max_tensor_idx);
        } else {
          //std::cout << "\t " << max_tensor_idx << " fail \n";
        }

        out_tensor[max_tensor_idx] = -9999999;
      }
      // Pytorch tensor supports int64_t index, but Parla
      // chooses a device ID and so casting to uint32_t is fine.
      //std::cout << "\t fail:" << max_tensor_idx << "\n";
      return static_cast<DevID_t>(max_tensor_idx);
    } else {
      //std::cout << ">> Random: " << " \n";
      std::uniform_real_distribution<> devid_distribution(
          0.f, static_cast<float>(this->n_actions_));
      DevID_t randomly_chosen_device{1};
      for (size_t a = 0; a < this->n_actions_; ++a) {
        randomly_chosen_device =
            static_cast<DevID_t>(devid_distribution(mt));
        if (mask == nullptr || (mask != nullptr && (*mask)[randomly_chosen_device])) {
          return randomly_chosen_device;
        }
      }
      return randomly_chosen_device;
    }
  }

  void optimize_model() {
    if (!this->is_training_mode()) {
      return;
    }

    if (this->replay_memory_.size() < this->batch_size_) {
      return;
    }

    std::cout << this->episode_ << " optimization..\n";

    /*
    size_t p_i{0};
    std::ofstream fp_b(std::to_string(this->episode_) + ".before");
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      fp_b << p_i++ << ":" << parameter << "\n";
    }
    fp_b.close();
    */

    std::vector<BufferTupleType> batch =
        this->replay_memory_.sample(this->batch_size_);
    std::vector<torch::Tensor> curr_states;
    std::vector<torch::Tensor> next_states;
    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> rewards;

    // Stack Ss, S's, As, Rs in the batch to create 2D matrices.
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
    // Action was 1 dimensional tensor, so needs to be unsqueezed to be
    // a (1, :) dimension tensor.
    actions_tensor = torch::cat(actions, 0).to(this->device_).unsqueeze(0);
    rewards_tensor = torch::cat(rewards, 0).to(this->device_);

    // Calculate expected Q value.
    torch::Tensor q_values = this->policy_net_.forward(curr_states_tensor);
    // Gather q values corresponding to chosen actions.
    q_values = q_values.gather(1, actions_tensor).reshape({this->batch_size_, 1ULL});
    torch::Tensor next_target_q_values =
        this->target_net_.forward(next_states_tensor);
    //std::cout << "next Q values:" << next_target_q_values << "\n";
    //std::cout << "next Q max values:" << std::get<0>(next_target_q_values.max(1)) << "\n";
    next_target_q_values = std::get<0>(next_target_q_values.max(1));
    torch::Tensor expected_q_values =
        this->gamma_ * next_target_q_values + rewards_tensor;
    //std::cout << " qvals:" << q_values << ", "
    //          << " action_tensor:" << actions_tensor << ", "
    //          << " action_tensor unsqueezed:" << actions_tensor.unsqueeze(1)
    //          << ", reward tensor:" << rewards_tensor << ", "
    //          << ", expected q values:" << expected_q_values << "\n";

    torch::Tensor loss = torch::smooth_l1_loss(q_values, expected_q_values.reshape({this->batch_size_, 1ULL}));
    // Zerofying gradients in the optimizer.
    this->rms_optimizer_.zero_grad();
    // Update gradients.
    loss.backward();
    std::ofstream fp("loss.out", std::ios_base::app);
    fp << loss.item<float>() << "\n";
    fp.close();
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      parameter.grad().data().clamp(-1, 1);
    }
    this->rms_optimizer_.step();
    /*
    p_i = 0;
    std::ofstream fp_a(std::to_string(this->episode_) + ".after");
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      fp_a << p_i++ << ":" << parameter << "\n";
    }
    fp_a.close();
    */
  }

  void target_net_soft_update(float TAU = 0.005) {
    torch::NoGradGuard no_grad;
    auto target_named_parameters = this->target_net_.named_parameters();
    auto named_parameters = this->policy_net_.named_parameters(true);
    auto named_buffers = this->policy_net_.named_buffers(true);
    for (auto &named_parameter : named_parameters) {
      std::string param_key = named_parameter.key();
      torch::Tensor param_value = named_parameter.value();
      torch::Tensor *target_param_val_ptr = target_named_parameters.find(param_key);
      if (target_param_val_ptr != nullptr) {
        /*
        std::cout << param_key << ", " << named_parameter.value() 
          << ", old " << *target_param_val_ptr << "\n";
        */
        torch::Tensor new_param_val =
            param_value * TAU + *target_param_val_ptr * (1 - TAU);
        target_param_val_ptr->copy_(new_param_val);
      } else {
        target_param_val_ptr = named_buffers.find(param_key);
        if (target_param_val_ptr != nullptr) {
          /*
          std::cout << param_key << ", " << named_parameter.value() 
             << ", old " << *target_param_val_ptr << "\n";
          */
          torch::Tensor new_param_val =
              param_value * TAU + *target_param_val_ptr * (1 - TAU);
          target_param_val_ptr->copy_(new_param_val);
        }
      }
      /*
      std::cout << param_key << ", " << named_parameter.value() 
        << ", new " << *target_param_val_ptr << "\n";
      */
    }
  }

  void append_replay_memory(torch::Tensor curr_state,
                            torch::Tensor chosen_device,
                            torch::Tensor next_state, torch::Tensor reward) {
    this->replay_memory_.push(curr_state, chosen_device, next_state, reward,
                              this->episode_);
  }

  void print() { this->replay_memory_.print(); }

  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, uint64_t>>
  sample(int64_t b) {
    return this->replay_memory_.sample(b);
  }

  void incr_episode() {
    ++this->episode_;
  }

  size_t get_episode() {
    return this->episode_;
  }

  size_t get_replay_memory_size() {
    return this->replay_memory_.size();
  }

  bool is_training_mode() {
    if (this->rl_mode_.compare("training") == 0) {
      return true;
    } else {
      return false;
    }
  }

private:
  // TODO: replay memory

  FullyConnectedDQNImpl policy_net_;
  FullyConnectedDQNImpl target_net_;
  torch::Device device_;
  std::string rl_mode_;
  uint32_t n_actions_;
  float eps_start_, eps_end_, eps_decay_;
  size_t batch_size_;
  float gamma_;
  uint64_t steps_;
  ExperienceReplay replay_memory_;
  torch::optim::RMSprop rms_optimizer_;
  size_t episode_;
};

class RLTaskMappingPolicy : public MappingPolicy {
public:
  RLTaskMappingPolicy(
      DeviceManager *device_manager, PArrayTracker *parray_tracker,
      Mapper *mapper);

  ~RLTaskMappingPolicy();

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
