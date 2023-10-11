import math
import random
import torch
import torch.nn
import torch.optim as optim

from ..networks.fcn import *

class DQNAgent:

    def __init__(self, in_dim: int, out_dim: int,
                 eps_start = 0.9, eps_end = 0.05, eps_decay = 1000,
                 batch_size = 516, gamma = 0.999):
        self.policy_network = FCN(in_dim, out_dim)
        self.target_network = FCN(in_dim, out_dim)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(),
                                       lr=0.002)
        # RL parameter setup
        self.n_actions = out_dim
        self.steps = 1
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma

    def select_device(self, x):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps / self.eps_decay)
        self.steps += 1
        if sample > eps_threshold:
            print("Policy network is used", flush=True)
            # TODO(hc): Always use this condition if test mode is enabled
            with torch.no_grad():
                out = self.policy_network(x)
                for action in range(self.n_actions):
                    max_action_pair = out.max(0)
                    # TODO(hc): may need to check if max action is within
                    # resource available devices.
                    return max_action_pair[1]
        else:
            print("Random generator is used", flush=True)
            # TODO(hc): Check mask; the mask is marked if that device does not
            # have sufficient resources.
            out = torch.tensor(
                  random.choice([d for d in range(self.n_actions)]),
                  dtype=torch.float)

        return out

    def optimize_model(self):
        # TODO(hc): Check if the current mode is the training mode.
        # TODO(hc): Check if the current replay memory size is less than the
        #           batch size.
        # TODO(hc): Then, do optimization.
        pass

    def update_target_network(self):
        pass

    def load_networks(self):
        pass

    def save_networks(self):
        pass

    def load_optimizer(self):
        pass

    def save_optimizer(self):
        pass
