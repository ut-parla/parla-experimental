#ifndef PARLA_RL_TASK_MAPPER
#define PARLA_RL_TASK_MAPPER

#include "device_manager.hpp"
#include "rl_environment.hpp"
#include "rl_utils.h"
#include "runtime.hpp"
#include "policy.hpp"

#include <random>
#include <torch/torch.h>

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
    this->mtx_.lock();
    while (this->buffer_.size() >= this->capacity_) {
      this->buffer_.pop_front();
    }
    this->buffer_.push_back(new_buffer_element);
    this->mtx_.unlock();
  }

  /*
   * Sample tuples (S, S', R, A) from the experience replay buffer.
   *
   * @param batch_size Batch size
   */
  std::vector<BufferTupleType> sample(int64_t batch_size) {
    this->mtx_.lock();
    std::vector<BufferTupleType> sampled_buffer(batch_size);
    std::sample(this->buffer_.begin(), this->buffer_.end(),
                sampled_buffer.begin(), sampled_buffer.size(),
                std::mt19937_64{std::random_device{}()});
    this->mtx_.unlock();
    return sampled_buffer;
  }

  /*
   * Return the number of tuples in the experience replay memory.
   */
  size_t size() {
    size_t sz{0};
    this->mtx_.lock();
    sz = this->buffer_.size();
    this->mtx_.unlock();
    return sz;
  }

  void clear() {
    this->mtx_.lock();
    this->buffer_.clear();
    this->mtx_.unlock();
  }

  void print() {
    this->mtx_.lock();
    for (size_t i = 0; i < this->buffer_.size(); ++i) {
      std::cout << "\n [" << i << "th buffer]\n";
      std::cout << "\t Current state: " << std::get<0>(this->buffer_[i]) << ", "
                << "Chosen device: " << std::get<1>(this->buffer_[i]) << ", "
                << "Next state: " << std::get<2>(this->buffer_[i]) << ", "
                << "Reward: " << std::get<3>(this->buffer_[i]) << "\n";
    }
    this->mtx_.unlock();
  }

private:
  /// The maximum experience replay memory size.
  int64_t capacity_;
  /// Experience replay memory.
  BufferTy buffer_;
  std::mutex mtx_;
};

class RLStateTransition {
  public:
    std::string task_name;
    torch::Tensor current_state;
    torch::Tensor next_state;
    torch::Tensor chosen_device;
    torch::Tensor reward;
};

struct FullyConnectedDQNImpl : public torch::nn::Module {
  FullyConnectedDQNImpl(size_t in_dim, size_t out_dim)
      : in_dim_(in_dim), out_dim_(out_dim), device_(torch::kCUDA) {
    // Move networks to a GPU.
    fc1_ = torch::nn::Linear(in_dim, in_dim * 4);
    fc1_->to(torch::kDouble);
    fc1_ = register_module("fc1", fc1_);
    fc1_->to(device_);
    //batch_norm1_ = torch::nn::BatchNorm1d(in_dim * 4);
    //batch_norm1_->to(device_);
    fc2_ = torch::nn::Linear(in_dim * 4, in_dim * 8);
    fc2_->to(torch::kDouble);
    fc2_ = register_module("fc2", fc2_);
    fc2_->to(device_);
    //batch_norm2_ = torch::nn::BatchNorm1d(in_dim * 8);
    //batch_norm2_->to(device_);
    out_ = torch::nn::Linear(in_dim * 8, out_dim);
    out_->to(torch::kDouble);
    out_ = register_module("out", out_);
    out_->to(device_);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.to(device_);
    x = torch::nn::functional::normalize(x.view({x.size(0), in_dim_}));
    //std::cout << "f0:" << x << "\n";
    //x = torch::leaky_relu(torch::nn::functional::normalize(fc1_(x.view({x.size(0), in_dim_}))));
    x = torch::leaky_relu(fc1_(x));
    //x = torch::leaky_relu(fc1_(x.view({x.size(0), in_dim_})));
    //std::cout << "f1:" << x << "\n";
    //x = torch::leaky_relu(batch_norm2_(fc2_(x)));
    //x = torch::leaky_relu(torch::nn::functional::normalize(fc2_(x)));
    x = torch::leaky_relu(fc2_(x));
    //std::cout << "f2:" << x << "\n";
    //std::cout << "f3:" << x << "\n";
    //x = out_(x);
    //x = torch::log_softmax(torch::nn::functional::normalize(out_(x)), 1);
    x = torch::log_softmax(out_(x), 1);
    std::cout << "out:" << x << "\n";
    return x.squeeze(0);
  }

  uint32_t in_dim_, out_dim_;
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, out_{nullptr};
  //torch::nn::BatchNorm1d batch_norm1_{nullptr}, batch_norm2_{nullptr};
  torch::Device device_;
};

//TORCH_MODULE(FullyConnectedDQN);

class RLAgent {
public:
  using BufferTupleType = typename ExperienceReplay::BufferTupleType;


  RLAgent(size_t in_dim, size_t out_dim, uint32_t n_actions, bool is_training_mode = true,
          torch::Device device = torch::kCUDA,
          float eps_start = 0.9, float eps_end = 0.05, float eps_decay = 1000,
          size_t batch_size = 516, float gamma = 0.999)
      : policy_net_(in_dim, out_dim), target_net_(in_dim, out_dim),
        device_(device), n_actions_(n_actions), is_training_mode_(is_training_mode),
        eps_start_(eps_start), eps_end_(eps_end), eps_decay_(eps_decay),
        batch_size_(batch_size), gamma_(gamma), steps_(0),
        replay_memory_(10000),
        rms_optimizer_(policy_net_.parameters(), torch::optim::RMSpropOptions(0.002)),
        //adam_optimizer_(policy_net_.parameters(), torch::optim::AdamOptions(3e-4)),
        episode_(0) {
    this->load_models();      
  }

  ~RLAgent() {
    if (this->is_training_mode()) {
      this->save_models();
    }
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
    //torch::save(this->adam_optimizer_, "adam_optimizer.pt");
  }

  void load_models() {
    std::cout << "Load models..\n";
    if (std::ifstream fp("policy_net.pt"); fp) {
      std::cout << "Load policy net\n";
      torch::serialize::InputArchive input_archive;
      input_archive.load_from("policy_net.pt");
      this->policy_net_.load(input_archive);
    }
    if (std::ifstream fp("target_net.pt"); fp) {
      std::cout << "Load target net\n";
      torch::serialize::InputArchive input_archive;
      input_archive.load_from("target_net.pt");
      this->target_net_.load(input_archive);
    } else {
      std::cout << "Update target net parameter..\n";
      torch::NoGradGuard no_grad;
      auto target_named_parameters = this->target_net_.named_parameters();
      auto named_parameters = this->policy_net_.named_parameters(true);
      auto named_buffers = this->policy_net_.named_buffers(true);
      for (auto &named_parameter : named_parameters) {
        std::string param_key = named_parameter.key();
        torch::Tensor param_value = named_parameter.value();
        torch::Tensor *target_param_val_ptr = target_named_parameters.find(param_key);
        if (target_param_val_ptr != nullptr) {
          target_param_val_ptr->copy_(param_value);
        } else {
          target_param_val_ptr = named_buffers.find(param_key);
          if (target_param_val_ptr != nullptr) {
            target_param_val_ptr->copy_(param_value);
          }
        }
      }
    }
    
    if (std::ifstream fp("rms_optimizer.pt"); fp) {
      std::cout << "Load ADAM optimizer\n";
      torch::load(this->rms_optimizer_, "rms_optimizer.pt");
      //torch::load(this->adam_optimizer_, "adam_optimizer.pt");
    }
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
      torch::Tensor out_tensor = this->policy_net_.forward(state).clone();
      int64_t max_tensor_idx{1};
      for (size_t a = 0; a < this->n_actions_; ++a) {
        auto max_action_pair = out_tensor.max(0);
        torch::Tensor max_tensor = std::get<0>(max_action_pair);
        max_tensor_idx = (std::get<1>(max_action_pair)).item<int64_t>();
        if (mask == nullptr || (mask != nullptr && (*mask)[max_tensor_idx])) {
          //std::cout << "State:" << state << "\n";
          //std::cout << "Output:" << out_tensor << "\n";
          //std::cout << "\t" << max_tensor_idx << "\n";
          return static_cast<DevID_t>(max_tensor_idx);
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
    actions_tensor = torch::cat(actions, 0).to(this->device_).unsqueeze(1);
    rewards_tensor = torch::cat(rewards, 0).to(torch::kFloat).to(this->device_);

    // Calculate expected Q value.
    torch::Tensor q_values = this->policy_net_.forward(curr_states_tensor);
    // Gather q values corresponding to chosen actions.
    q_values = q_values.gather(1, actions_tensor);
    torch::Tensor next_target_q_values =
        this->target_net_.forward(next_states_tensor);
    next_target_q_values = std::get<0>(next_target_q_values.max(1)).detach();
    torch::Tensor expected_q_values =
        this->gamma_ * next_target_q_values + rewards_tensor;
#if 0
    std::cout << "curr_states_tensor:" << curr_states_tensor << "\n";
    std::cout << "rewards:" << rewards_tensor << "\n";
    std::cout << "action:" << actions_tensor.squeeze(0) << "\n";
    std::cout << " qvals:" << q_values << ", "
              << " action_tensor:" << actions_tensor << ", "
              << " action_tensor unsqueezed:" << actions_tensor.unsqueeze(1)
              << ", reward tensor:" << rewards_tensor << ", "
              << ", expected q values:" << expected_q_values << "\n";
#endif
    torch::Tensor loss = torch::smooth_l1_loss(q_values, expected_q_values.unsqueeze(1));
    std::cout << "\n Loss:" << loss << "\n";

    // Zerofying gradients in the optimizer.
    this->rms_optimizer_.zero_grad();
    //this->adam_optimizer_.zero_grad();
    // Update gradients.
    loss.backward();
    this->rms_optimizer_.step();
    //this->adam_optimizer_.step();
    std::cout << "optimization step done\n";
  }

  void target_net_soft_update_simpler(float TAU = 0.005) {
    if (this->episode_ > 0 && this->episode_ % 3 == 0) {
      return;
    }
    for (auto &p : this->target_net_.named_parameters()) {
      torch::NoGradGuard no_grad;
      p.value().copy_(
          TAU * p.value() +
          (1 - TAU) * this->policy_net_.named_parameters()[p.key()]);
    }
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
        torch::Tensor new_param_val =
            param_value * TAU + *target_param_val_ptr * (1 - TAU);
        target_param_val_ptr->copy_(new_param_val);
      } else {
        target_param_val_ptr = named_buffers.find(param_key);
        if (target_param_val_ptr != nullptr) {
          torch::Tensor new_param_val =
              param_value * TAU + *target_param_val_ptr * (1 - TAU);
          target_param_val_ptr->copy_(new_param_val);
        }
      }
    }
  }

  void append_replay_memory(torch::Tensor curr_state,
                            torch::Tensor chosen_device,
                            torch::Tensor next_state, torch::Tensor reward) {
    this->replay_memory_.push(curr_state, chosen_device, next_state, reward,
                              this->episode_);
  }

  void append_mapped_task_info(
      InnerTask *task, torch::Tensor current_state, torch::Tensor next_state,
      torch::Tensor chosen_device, torch::Tensor reward) {
    if (!check_valid_tasks(task->name)) {
      return;
    }
    RLStateTransition *tinfo = new RLStateTransition();
    tinfo->task_name = task->name;
    tinfo->current_state = current_state;
    tinfo->next_state = next_state;
    tinfo->chosen_device = chosen_device;
    tinfo->reward = reward;
    this->replay_mem_buffer_mtx_.lock();
    task->replay_mem_buffer_id_ = this->replay_memory_buffer_.size();
    this->replay_memory_buffer_.push_back(tinfo);
    this->replay_mem_buffer_mtx_.unlock();
  }

  void evaluate_current_epoch(double exec_time_ms, double best_exec_time_ms_exec_time_ms) {
#if 0
    std::cout << "Evaluate current epoch:" << exec_time_ms << " vs " <<
      best_exec_time_ms_exec_time_ms << "\n";
#endif
    if (this->is_training_mode_) {
      bool is_worth = (exec_time_ms <= best_exec_time_ms_exec_time_ms);
      this->replay_mem_buffer_mtx_.lock();
      double positive_weight = (exec_time_ms == 0)? 0 : (best_exec_time_ms_exec_time_ms / exec_time_ms);
      double negative_weight = (best_exec_time_ms_exec_time_ms == 0)? 0 : (exec_time_ms / best_exec_time_ms_exec_time_ms);
      double constant =
        (exec_time_ms == 0 ||
        best_exec_time_ms_exec_time_ms == std::numeric_limits<double>::max())?
            0 : (best_exec_time_ms_exec_time_ms / exec_time_ms);
      if (constant < 0.8) {
        constant = 0; 
      }
      for (size_t i = 0; i < this->replay_memory_buffer_.size(); ++i) {
        RLStateTransition *tinfo = this->replay_memory_buffer_[i];
        // All (S, A, R)s get constant reward if the current episode 
        // is fast.
        torch::Tensor final_reward =
            torch::tensor({tinfo->reward.item<double>() + constant}, torch::kDouble);
        this->append_replay_memory(
            tinfo->current_state, tinfo->chosen_device, tinfo->next_state,
            final_reward);
      }
      this->replay_mem_buffer_mtx_.unlock();
    }
    this->optimize_model();
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
    return this->is_training_mode_;
  }

  void clear_replay_memory_buffer() {
    this->replay_mem_buffer_mtx_.lock();
    this->replay_memory_buffer_.clear();
    this->replay_mem_buffer_mtx_.unlock();
  }

private:
  // TODO: replay memory

  FullyConnectedDQNImpl policy_net_;
  FullyConnectedDQNImpl target_net_;
  torch::Device device_;
  uint32_t n_actions_;
  bool is_training_mode_;
  float eps_start_, eps_end_, eps_decay_;
  size_t batch_size_;
  float gamma_;
  uint64_t steps_;
  ExperienceReplay replay_memory_;
  torch::optim::RMSprop rms_optimizer_;
  //torch::optim::Adam adam_optimizer_;
  size_t episode_;
  size_t subepisode_{0};
  std::vector<RLStateTransition*> replay_memory_buffer_;
  std::mutex replay_mem_buffer_mtx_;
};

class RLTaskMappingPolicy : public MappingPolicy {
public:
  RLTaskMappingPolicy(
      DeviceManager *device_manager, PArrayTracker *parray_tracker,
      InnerScheduler *sched, bool is_training_mode);

  ~RLTaskMappingPolicy();

  bool calc_score_devplacement(
      InnerTask *task,
      const std::shared_ptr<DeviceRequirement> &dev_placement_req,
      InnerScheduler *sched, Score_t *score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
          &parray_list) override;

  bool calc_score_archplacement(
      InnerTask *task, ArchitectureRequirement *arch_placement_req,
      InnerScheduler *sched, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
      Score_t *chosen_dev_score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
          &parray_list,
      std::vector<bool> *is_dev_assigned = nullptr) override;

  bool calc_score_mdevplacement(
      InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
      InnerScheduler *sched,
      std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
      Score_t *average_score,
      const std::vector<
          std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list) override;

  void run_task_mapping(
      InnerTask *task, InnerScheduler *sched,
      std::vector<std::shared_ptr<DeviceRequirement>> *chosen_devices,
      const std::vector<std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list,
      std::vector<std::shared_ptr<PlacementRequirementBase>>
          *placement_req_options_vec) override;

  void evaluate_current_epoch(double exec_time_ms, double previous_exec_time_ms) {
    this->rl_agent_->evaluate_current_epoch(exec_time_ms, previous_exec_time_ms);
    this->rl_agent_->incr_episode();
    this->rl_agent_->target_net_soft_update_simpler();
    this->rl_env_->output_reward(this->rl_agent_->get_episode());
    this->rl_agent_->clear_replay_memory_buffer();
    if (this->rl_agent_->get_episode() % 10 == 0 && this->rl_agent_->is_training_mode()) {
      std::cout << "Episode " << this->rl_agent_->get_episode() <<
          ": store models..\n";
      this->rl_agent_->save_models();
    }
  }

private:
  /// RL agent.
  RLAgent *rl_agent_;
  /// RL environment.
  RLEnvironment *rl_env_;
  torch::Tensor rl_current_state_;
  torch::Tensor rl_next_state_;
};

#endif
