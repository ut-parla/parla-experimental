import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from ..networks.gcn_fcn_leakyrelu import *
from .globals import *

class A2CAgent:

    def __init__(self, gcn_indim: int, in_dim: int, out_dim: int,
                 execution_mode: str = "training"):
        # Policy network that selects an action.
        self.actor = GCN_FCN_Type2(gcn_indim, in_dim, out_dim)
        # Value network that evaluates an action from the policy network.
        self.critic = GCN_FCN_Type2(gcn_indim, in_dim, 1)

        self.lst_log_probs = []
        self.lst_values = []
        self.lst_rewards = []
        self.entropy = 0
        self.steps = 0
        self.step_for_optim = 5

    def select_device(self, x: torch.tensor, gcn_x = None, gcn_edgeindex = None):
        with torch.no_grad():
            model_input = NetworkInput(x, False, gcn_x, gcn_edgeindex)
            actions = self.actor(model_input)
            value = torch.squeeze(self.critic(model_input))
            action_probabilities = F.softmax(actions, dim=-1)
            dist = Categorical(action_probabilities)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            reward = torch.tensor([0], dtype=torch.float)

            self.lst_log_probs.append(log_prob)
            self.lst_values.append(value)
            self.lst_rewards.append(0)

            self.entropy += dist.entropy().mean()
            self.steps += 1

            print("steps:", self.steps, " entropy:", self.entropy)
            if self.steps == self.step_for_optim:
                # 1. next state.
                # 2. next value = self.critic(next_state)
                cat_log_probs = torch.cat(self.lst_log_probs)
                cat_values = torch.cat(self.lst_values)
                # 3. Calculate advantages (G = reward + gamma * value) - values (Vs)

                # 4. actor loss = - (log_probs * advantage).mean()
                # 5. critic loss = advantage^2; since advantage is already TD, and so on MSE,
                # advantage^2.

                # 6. loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
                # 7. optimize model.
                self.steps = 0
                self.entropy = 0
                # 8. optimizer.zero_grad()
                # 9. loss.backward()
                # 10. optimizer.step()
            print("actions:", actions, " action probas:", action_probabilities, " action:", action, " value:", value)
            return action
        # If the current selection is < batch size, choose action and accumulate
        # (value, reward, log probability).
        # 1. Get a probabiliy of actions on the current state through actor. 
        # 2. Sample an action by using Categorical random distribution.
        # 3. Get log probability of the action over the 1's probabilities.
        # 4. Append (V, LP, R)
        #
        # If the current selection is >= batch size, calculate advantage value, and
        # update models.
        # 1. Calculate next state, and its next value from the critic network.

