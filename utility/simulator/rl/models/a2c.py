import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from ..networks.a2c_gcn_fcn import *
from .globals import *

class A2CAgent:
    # TODO(hc): add save/load netowrks.
    # TODO(hc): add training/testing phase modes.
    # TODO(hc): if testing mode is enabled, skip the model optimization.

    def __init__(self, gcn_indim: int, in_dim: int, out_dim: int,
                 execution_mode: str = "training", gamma: float = 0.999):
        # Actor: Policy network that selects an action.
        # Critic: Value network that evaluates an action from the policy network.
        self.a2c_model = A2CNetwork(gcn_indim, in_dim, out_dim)
        self.optimizer = optim.RMSprop(self.a2c_model.parameters(),
                                       lr=0.002)
        # Store log probabilty; later elements are concatenated to a tensor.
        self.lst_log_probs = []
        # Store values from the critic network; to be concatenated to a tensor.
        self.lst_values = []
        # Store rewards; to be concatenated to a tensor
        self.lst_rewards = []
        self.entropy = 0
        self.steps = 0
        # Interval to update the actor network parameter
        self.step_for_optim = 5
        self.gamma = gamma

    def construct_next_state(self, x: torch.tensor, gcn_x = None, gcn_edgeindex = None):
        x[0] = x[0] + 1
        return x, gcn_x, gcn_edgeindex

    def compute_returns(self, next_value, rewards):
        returns = []
        R = next_value
        # Rewards are stored in time sequence.
        # R on the deeper level should be used for more latest return
        # calculation.
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R
            # Restore the time sequence order.
            returns.insert(0, R)
        return returns

    def select_device(self, x: torch.tensor, gcn_x = None, gcn_edgeindex = None):
        # Different from DQN, A2C requires gradient tracking.
        model_input = NetworkInput(x, False, gcn_x, gcn_edgeindex)
        # This gets two values:
        # 1) transition probability of all the actions from the current state
        # 2) state value that evaluates the actor's policy;
        #    if a state value and probability distribution are corresponding,
        #    it is a good model.
        actions, value = self.a2c_model(model_input)
        print("model input:", model_input)
        print("gcn input:", gcn_x)
        print("gcn edgeindex:", gcn_edgeindex)
        print("actions:", actions)
        print("value:", value)
        action_probabilities = F.softmax(actions, dim=0)
        # Sample an action by using Categorical random distribution
        dist = Categorical(action_probabilities)
        action = dist.sample()
        # Get log probability of the action
        log_prob = dist.log_prob(action)
        # TODO(hc): replace it with the actual reward.
        reward = torch.tensor([10], dtype=torch.float)

        print("action porbs:", action_probabilities)
        print("log prob:", log_prob)
        print("reward:", reward)

        self.lst_log_probs.append(log_prob)
        self.lst_values.append(value)
        self.lst_rewards.append(reward)

        assert len(self.lst_log_probs) <= self.step_for_optim 
        assert len(self.lst_values) <= self.step_for_optim 
        assert len(self.lst_rewards) <= self.step_for_optim 

        self.entropy += dist.entropy().mean()
        self.print_model(str(self.steps))
        self.steps += 1


        print("steps:", self.steps, " entropy:", self.entropy)
        # Update the actor network
        if self.steps == self.step_for_optim:
            assert len(self.lst_log_probs) == self.step_for_optim 
            assert len(self.lst_values) == self.step_for_optim 
            assert len(self.lst_rewards) == self.step_for_optim 

            # TODO(hc): replace it with the actual next state
            next_x, next_gcn_x, next_gcn_edgeindex = \
                self.construct_next_state(x, gcn_x, gcn_edgeindex)
            # To perform TD to optimize the model, get a state value
            # of the expected next state from the critic network
            _, next_value = self.a2c_model(
                NetworkInput(next_x, False, next_gcn_x, next_gcn_edgeindex))
            cat_log_probs = torch.cat(
                [lp.unsqueeze(0) for lp in self.lst_log_probs])
            cat_values = torch.cat([v for v in self.lst_values])
            cat_rewards = torch.cat([r for r in self.lst_rewards])
            returns = self.compute_returns(next_value, cat_rewards)
            returns = torch.cat(returns).detach()
            print("\t log probs:", cat_log_probs)
            print("\t values:", cat_values)
            print("\t rewards:", cat_rewards)
            print("\t return:", returns)
            print("\t next value:", next_value)
            print("\t next x:", next_x)

            advantage = returns - cat_values
            print("\t advantage:", advantage)

            actor_loss = -(cat_log_probs * advantage.detach())
            print("\t actor loss:", actor_loss)
            actor_loss = actor_loss.mean()
            print("\t actor loss mean:", actor_loss)

            critic_loss = advantage.pow(2).mean()
            print("\t critic loss:", critic_loss)

            loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy
            print("\t loss:", loss)
            self.steps = 0
            self.entropy = 0
            self.lst_log_probs = []
            self.lst_values = []
            self.lst_rewards = []

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.a2c_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.print_model("optimization")
        return action

    def print_model(self, prefix: str):
        with open("models/" + prefix + ".a2c_network.str", "w") as fp:
            for key, param in self.a2c_model.named_parameters():
                fp.write(key + " = " + str(param))
        with open("models/" + prefix + ".optimizer.str", "w") as fp:
            for key, param in self.optimizer.state_dict().items():
                fp.write(key + " = " + str(param))

