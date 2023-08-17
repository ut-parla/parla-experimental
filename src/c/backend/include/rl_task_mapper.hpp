#ifndef PARLA_RL_TASK_MAPPER
#define PARLA_RL_TASK_MAPPER

#include "device_manager.hpp"
#include "rl_environment.hpp"
#include "runtime.hpp"
#include "policy.hpp"

#include <random>
#include <torch/torch.h>

#define NUM_TASK_FEATURES 4
#define NUM_DEP_TASK_FEATURES 3
#define NUM_DEVICE_FEATURES 4
#define DEVICE_FEATURE_OFFSET (NUM_TASK_FEATURES + NUM_DEP_TASK_FEATURES * 2)

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

class RLStateTransition {
  public:
    torch::Tensor current_state;
    torch::Tensor chosen_device;
    double base_score{0};
};

struct FullyConnectedDQNImpl : public torch::nn::Module {
  FullyConnectedDQNImpl(size_t in_dim, size_t out_dim)
      : in_dim_(in_dim), out_dim_(out_dim), device_(torch::kCUDA) {
    // Move networks to a GPU.
    fc1_ = torch::nn::Linear(in_dim, in_dim * 4);
    fc1_->to(torch::kDouble);
    fc1_ = register_module("fc1", fc1_);
    fc1_->to(device_);
    fc2_ = torch::nn::Linear(in_dim * 4, in_dim * 4);
    fc2_->to(torch::kDouble);
    fc2_ = register_module("fc2", fc2_);
    fc2_->to(device_);
    out_ = torch::nn::Linear(in_dim * 4, out_dim);
    out_->to(torch::kDouble);
    out_ = register_module("out", out_);
    out_->to(device_);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.to(device_);
    x = torch::relu(fc1_(x.view({x.size(0), in_dim_})));
    x = torch::relu(fc2_(x));
    //x = out_(x);
    x = torch::log_softmax(out_(x), 1);
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


  RLAgent(size_t in_dim, size_t out_dim, uint32_t n_actions, bool is_training_mode = true,
          torch::Device device = torch::kCUDA,
          float eps_start = 0.9, float eps_end = 0.05, float eps_decay = 1000,
          size_t batch_size = 128, float gamma = 0.999)
      : policy_net_(in_dim, out_dim), target_net_(in_dim, out_dim),
        device_(device), n_actions_(n_actions), is_training_mode_(is_training_mode),
        eps_start_(eps_start), eps_end_(eps_end), eps_decay_(eps_decay),
        batch_size_(batch_size), gamma_(gamma), steps_(0),
        replay_memory_(10000),
        //rms_optimizer_(policy_net_.parameters(), torch::optim::RMSpropOptions(0.025)),
        adam_optimizer_(policy_net_.parameters(), torch::optim::AdamOptions(3e-4)),
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
    //torch::save(this->rms_optimizer_, "rms_optimizer.pt");
    torch::save(this->adam_optimizer_, "adam_optimizer.pt");
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
        //std::cout << param_key << ", " << named_parameter.value() 
        //  << ", new " << *target_param_val_ptr << "\n";
      }
    }
    
    if (std::ifstream fp("adam_optimizer.pt"); fp) {
      std::cout << "Load ADAM optimizer\n";
      //torch::load(this->rms_optimizer_, "rms_optimizer.pt");
      torch::load(this->adam_optimizer_, "adam_optimizer.pt");
    }
#if 0
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
      torch::Tensor out_tensor = this->policy_net_.forward(state).clone();
      //std::cout << ">> Out:" << out_tensor.unsqueeze(0) << "\n";
      int64_t max_tensor_idx{1};
      for (size_t a = 0; a < this->n_actions_; ++a) {
        auto max_action_pair = out_tensor.max(0);
        torch::Tensor max_tensor = std::get<0>(max_action_pair);
        max_tensor_idx = (std::get<1>(max_action_pair)).item<int64_t>();
        if (mask == nullptr || (mask != nullptr && (*mask)[max_tensor_idx])) {
          //std::cout << "\t" << max_tensor_idx << "\n";
          return static_cast<DevID_t>(max_tensor_idx);
        } else {
          std::cout << "\t " << max_tensor_idx << " fail \n";
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
    actions_tensor = torch::cat(actions, 0).to(this->device_).unsqueeze(1);
    rewards_tensor = torch::cat(rewards, 0).to(torch::kFloat).to(this->device_);

    // Calculate expected Q value.
    torch::Tensor q_values = this->policy_net_.forward(curr_states_tensor);
    //std::cout << "Before current Q values:" << q_values << "\n";
    //std::cout << "Before action:" << actions_tensor << "\n";
    // Gather q values corresponding to chosen actions.
    q_values = q_values.gather(1, actions_tensor);
    torch::Tensor next_target_q_values =
        this->target_net_.forward(next_states_tensor);
    //std::cout << "current Q values:" << curr_states_tensor << "\n";
    //std::cout << "next Q values:" << next_states_tensor << "\n";
    //std::cout << "next Q max values:" << std::get<0>(next_target_q_values.max(1)) << "\n";
    //std::cout << "Before next Q values:" << next_target_q_values << "\n";
    next_target_q_values = std::get<0>(next_target_q_values.max(1)).detach();
    //std::cout << "Max next Q values:" << next_target_q_values << "\n";
    //std::cout << "Reward values:" << rewards_tensor << "\n";
    torch::Tensor expected_q_values =
        this->gamma_ * next_target_q_values + rewards_tensor;
    //std::cout << next_target_q_values << " vs reward: " << rewards_tensor << " =\n";
    //std::cout << "Expected q values:" << expected_q_values.unsqueeze(0) << "\n";
    //std::cout << "q values:" << q_values << "\n";

#if 0
    std::cout << "curr_states_tensor:" << curr_states_tensor << "\n";
    std::cout << "rewards:" << rewards_tensor << "\n";
    std::cout << "action:" << actions_tensor.squeeze(0) << "\n";
#endif
    /*std::cout << " qvals:" << q_values << ", "
              << " action_tensor:" << actions_tensor << ", "
              << " action_tensor unsqueezed:" << actions_tensor.unsqueeze(1)
              << ", reward tensor:" << rewards_tensor << ", "
              << ", expected q values:" << expected_q_values << "\n";
    */
    //std::cout << "q values:" << q_values << "\n";
    //std::cout << "expectedq vals:" << expected_q_values.unsqueeze(1) <<"\n";
    torch::Tensor loss = torch::smooth_l1_loss(q_values, expected_q_values.unsqueeze(1));
    //torch::Tensor loss = torch::mse_loss(q_values, expected_q_values.unsqueeze(1));
    std::cout << "\n Loss:" << loss << "\n";

    // Zerofying gradients in the optimizer.
    //this->rms_optimizer_.zero_grad();
    this->adam_optimizer_.zero_grad();
    /*
    for (torch::Tensor parameter : this->rms_optimizer_.parameters()) {
      std::cout << "After zerofying parameter:" << parameter.grad() << "\n";
    }
    */

    // Update gradients.
    loss.backward();
    std::ofstream fp("loss.out", std::ios_base::app);
    fp << this->episode_ << ", " << loss.item<float>() << "\n";
    fp.close();
    /*
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      std::cout << "Before parameter:" << parameter.grad().data() << "\n";
      parameter.grad().data() =
        parameter.grad().data().clamp(-1, 1);
      //std::cout << "After parameter:" << parameter.grad().data() << "\n";
    }
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      std::cout << "Before parameter:" << parameter << "\n";
    }
    torch::nn::utils::clip_grad_norm_(this->policy_net_.parameters(), 100);
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      std::cout << "After parameter gradient:" << parameter.grad() << "\n";
    }
    */

    /*
    for (torch::Tensor parameter : this->rms_optimizer_.parameters()) {
      std::cout << "Before optiimzer parameter:" << parameter.grad() << "\n";
    }
    */

    //this->rms_optimizer_.step();
    this->adam_optimizer_.step();
    /*
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      std::cout << "After parameter:" << parameter.grad().data() << "\n";
    }

    for (torch::Tensor parameter : this->rms_optimizer_.parameters()) {
      std::cout << "After optiimzer parameter:" << parameter.grad() << "\n";
    }
    */


    /*
    size_t p_i = 0;
    std::ofstream fp_a(std::to_string(this->episode_) + "-" + std::to_string(this->subepisode_) + ".after");
    for (torch::Tensor parameter : this->policy_net_.parameters()) {
      fp_a << p_i++ << ":" << parameter << "\n";
    }
    fp_a.close();
    this->subepisode_ += 1;
    */
  }

  void target_net_soft_update_simpler(float TAU = 0.005) {
    for (auto &p : this->target_net_.named_parameters()) {
      torch::NoGradGuard no_grad;
      p.value().copy_(
          TAU * p.value() +
          (1 - TAU) * this->policy_net_.named_parameters()[p.key()]);
    }
  }

  void target_net_soft_update(float TAU = 0.005) {
    /*
    if (this->episode_ % 500 == 0) {
      return;
    }
    */
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

  void append_mapped_task_info(
      InnerTask *task, torch::Tensor current_state,
      torch::Tensor chosen_device) {
    if (task->name.find("global_0") != std::string::npos ||
        task->name.find("begin_rl_task") != std::string::npos ||
        task->name.find("end_rl_task") != std::string::npos) {
      return;
    }
    //auto current_time = std::chrono::high_resolution_clock::now();
    RLStateTransition *tinfo = new RLStateTransition();
    tinfo->current_state = current_state;
    tinfo->chosen_device = chosen_device;
    tinfo->base_score = (current_state[0][
        DEVICE_FEATURE_OFFSET +
        chosen_device.item<int64_t>() *
        NUM_DEVICE_FEATURES].item<double>() == 0)? 1 : 0;
    task->replay_mem_buffer_id_ = this->replay_memory_buffer_.size();
    this->replay_memory_buffer_.push_back(tinfo);
  }

  /**
   * @brief Calculate reward for a launched task and add the information
   * to the replay memory.
   *
   * @detail Task mapping information for a task is partially constructed
   * at a mapping phase and is stored in a temporary buffer.
   * When this task is about to be launched, calculate a reward with the
   * information in the buffer, and add it to the replay memory.
   *
   * @param task Inner task to register to the replay memory
   * @param rl_env RL environment having the replay memory
   */
  void evaluate_and_append_task_mapping(InnerTask *task, RLEnvironment *rl_env) {
    if (this->is_training_mode_) {
      if (task->name.find("global_0") != std::string::npos ||
          task->name.find("begin_rl_task") != std::string::npos ||
          task->name.find("end_rl_task") != std::string::npos) {
        return;
      }
      if (task->is_data_task()) { return; }
      // Get id of the task
      RLStateTransition *tinfo = this->replay_memory_buffer_[
          task->replay_mem_buffer_id_];
      DevID_t chosen_device_id = static_cast<DevID_t>(
          tinfo->chosen_device[0].item<int64_t>());
      torch::Tensor reward = rl_env->calculate_reward2(
            chosen_device_id, task, tinfo->current_state, tinfo->base_score);
      torch::Tensor next_state = rl_env->make_current_state(task);
      this->append_replay_memory(
          tinfo->current_state, tinfo->chosen_device, next_state, reward);
      this->optimize_model();
      this->target_net_soft_update_simpler();
      std::cout << this->get_episode() << " episode task " << task->name <<
          " current state:" << tinfo->current_state << ", device id:" <<
          tinfo->chosen_device << ", reward:" << reward << "\n";
    }
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
    this->replay_memory_buffer_.clear();
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
  //torch::optim::RMSprop rms_optimizer_;
  torch::optim::Adam adam_optimizer_;
  size_t episode_;
  size_t subepisode_{0};
  std::vector<RLStateTransition*> replay_memory_buffer_;
};

class RLTaskMappingPolicy : public MappingPolicy {
public:
  RLTaskMappingPolicy(
      DeviceManager *device_manager, PArrayTracker *parray_tracker,
      Mapper *mapper, bool is_training_mode);

  ~RLTaskMappingPolicy();

  bool calc_score_devplacement(
      InnerTask *task,
      const std::shared_ptr<DeviceRequirement> &dev_placement_req,
      Mapper *mapper, Score_t *score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
          &parray_list) override;

  bool calc_score_archplacement(
      InnerTask *task, ArchitectureRequirement *arch_placement_req,
      Mapper *mapper, std::shared_ptr<DeviceRequirement> &chosen_dev_req,
      Score_t *chosen_dev_score,
      const std::vector<std::pair<parray::InnerPArray *, AccessMode>>
          &parray_list,
      std::vector<bool> *is_dev_assigned = nullptr) override;

  bool calc_score_mdevplacement(
      InnerTask *task, MultiDeviceRequirements *mdev_placement_req,
      Mapper *mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *member_device_reqs,
      Score_t *average_score,
      const std::vector<
          std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list) override;

  void run_task_mapping(
      InnerTask *task, Mapper *mapper,
      std::vector<std::shared_ptr<DeviceRequirement>> *chosen_devices,
      const std::vector<std::vector<std::pair<parray::InnerPArray *, AccessMode>>>
          &parray_list,
      std::vector<std::shared_ptr<PlacementRequirementBase>>
          *placement_req_options_vec) override;

  /**
   * @brief Calculate reward for a launched task and add the information
   * to the replay memory.
   *
   * @detail Task mapping information for a task is partially constructed
   * at a mapping phase and is stored in a temporary buffer.
   * When this task is about to be launched, calculate a reward with the
   * information in the buffer, and add it to the replay memory.
   *
   * @param task Inner task to register to the replay memory
   */
  void evaluate_and_append_task_mapping(InnerTask *task) {
    mtx.lock();
    this->rl_agent_->evaluate_and_append_task_mapping(task, this->rl_env_);
    mtx.unlock();
  }

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
  //torch::Tensor rl_next_state_;
  std::mutex mtx;
};

#endif
