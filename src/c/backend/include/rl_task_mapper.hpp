#ifndef PARLA_RL_TASK_MAPPER
#define PARLA_RL_TASK_MAPPER

#include <torch/torch.h>

struct FullyConnectedDQN : torch::nn::Module {
  FullyConnectedDQN(uint32_t in_dim, uint32_t out_dim) :
      in_dim_(in_dim), out_dim_(out_dim), device_(torch::kCUDA) {
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
    x = torch::relu(fc2_->forward(x.reshape({in_dim_ * 4, in_dim_ * 4})));
    x = out_->forward(x.reshape({in_dim_ * 4, out_dim_}));
    return x;
  }

  uint32_t in_dim_, out_dim_;
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, out_{nullptr};
  torch::Device device_;
};

#endif
