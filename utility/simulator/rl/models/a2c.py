from torch.distributions import Categorical

from typing import Dict, List, Tuple
from collections import namedtuple

from ..networks.a2c_gcn_fcn import *
from ...task import SimulatedTask
from ....types import TaskState, TaskType
from .globals import *
from .model import *

import torchviz

# TODO(hc): This is deprecated as the reward system becomes an immediate reward.
A2CTransition = namedtuple("A2CTransition",
                          ("state", "action", "next_state", "reward",
                           # We may store information for a GCN layer too.
                           # If any GCN layer is not used, these should be `None`.
                           "gcn_state", "gcn_edgeindex", "next_gcn_state",
                           "next_gcn_edgeindex"))

class A2CAgent(RLModel):
    # TODO(hc): if testing mode is enabled, skip the model optimization.

    def __init__(self, gcn_indim: int, fcn_indim: int, outdim: int,
                 execution_mode: str = "training", gamma: float = 0.999):
        self.gcn_indim = gcn_indim
        self.indim = fcn_indim + gcn_indim
        self.outdim = outdim

        # Actor: Policy network that selects an action.
        # Critic: Value network that evaluates an action from the policy network.
        self.a2c_model = A2CNetwork(gcn_indim, self.indim, outdim)
        self.optimizer = optim.RMSprop(self.a2c_model.parameters(),
                                       lr=0.0001)
                                       #lr=0.0005)
        self.steps = 0
        self.execution_mode = execution_mode
        # Interval to update the actor network parameter
        self.step_for_optim = 10
        self.gamma = gamma
        self.episode = 0
        self.a2cnet_fname = "a2c_network.pt"
        self.optimizer_fname = "optimizer.pt"
        self.best_a2cnet_fname = "best_a2c_network.pt"
        self.best_optimizer_fname = "best_optimizer.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_mapping_decision = dict()
        # Accumulated reward 
        self.accumulated_reward = 0
        # Log action selection
        self.entropy_sum = 0
        self.log_probs = []
        self.values = []
        self.rewards = []

        if not self.is_training_mode():
            self.load_model()

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

    def select_device(self, target_task: SimulatedTask, x: torch.tensor,
                      gcn_x = None, gcn_edgeindex = None):
        # Different from DQN, A2C requires gradient tracking.
        model_input = NetworkInput(x, False, gcn_x, gcn_edgeindex)
        # This gets two values:
        # 1) transition probability of all the actions from the current state
        # 2) state value that evaluates the actor's policy;
        #    if a state value and probability distribution are corresponding,
        #    it is a good model.
        actions, value = self.a2c_model(model_input)
        action_probabilities = F.softmax(actions, dim=0)
        # Sample an action by using Categorical random distribution
        dist = Categorical(action_probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.entropy_sum += dist.entropy().mean()
        self.log_probs.append(log_prob)
        self.values.append(value)
        print("Target task:", target_task)
        print("action:", action)
        print("action probs:", action_probabilities)
        """
        print("select device:", target_task)
        print("model input:", model_input)
        print("gcn input:", gcn_x)
        print("gcn edgeindex:", gcn_edgeindex)
        print("value:", value)
        print("actions:", actions)
        print("entropy:", dist.entropy())
        print("sum:", self.entropy_sum)
        """
        return action

    def add_reward(self, reward):
        """
        Add a reward to the list.
        """
        self.accumulated_reward += reward.item()
        self.rewards.append(reward)

    def optimize_model(self, next_x, next_gcn_x, next_gcn_edgeindex):
        self.steps += 1
        if self.episode == 1 or not self.is_training_mode():
            # Reset the model states
            print("optimization ignored")
            self.steps = 0
            self.entropy_sum = 0
            self.log_probs = []
            self.values = []
            self.rewards = []
            return
        if self.steps == self.step_for_optim:
            print("optimization started")
            assert len(self.log_probs) == self.step_for_optim
            assert len(self.values) == self.step_for_optim
            assert len(self.rewards) == self.step_for_optim

            # To perform TD to optimize the model, get a state value
            # of the expected next state from the critic netowrk
            _, next_value = self.a2c_model(
                NetworkInput(next_x, False, next_gcn_x, next_gcn_edgeindex))
            cat_log_probs = torch.cat(
                [lp.unsqueeze(0) for lp in self.log_probs])
            cat_values = torch.cat(self.values)
            cat_rewards = torch.cat(self.rewards).to(self.device)
            returns = self.compute_returns(next_value, cat_rewards)
            returns = torch.cat(returns).detach() 
            advantage = returns - cat_values
            actor_loss = -(cat_log_probs * advantage.detach()).mean()
            #critic_loss = advantage.pow(2).mean()
            critic_loss = 1 * F.mse_loss(cat_values.unsqueeze(-1), returns.unsqueeze(-1))
            loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy_sum
            self.optimizer.zero_grad()
            loss.backward()
            # torchviz.make_dot(loss, params=dict(self.a2c_model.named_parameters())).render("attacehd", format="png")
            for param in self.a2c_model.parameters():
               param.grad.data.clamp_(-1, 1)
            """
            print("next x:", next_x)
            print("next_gcn_x:", next_gcn_x)
            print("next gcn edgeindex:", next_gcn_edgeindex)
            print("next value:", next_value)
            print("lst_log_probs:", cat_log_probs)
            print("lst rewards:", cat_rewards)
            print("lst values:", cat_values)
            print("cat returns:", returns)
            print("actor loss:", actor_loss, ", and critic loss:", critic_loss, " advantage:", advantage)
            print("loss;", loss)
            """
            # Reset the model states
            self.steps = 0
            self.optimizer.step()
            self.entropy_sum = 0
            self.log_probs = []
            self.values = []
            self.rewards = []

    def load_model(self):
        """ Load a2c model and optimizer parameters from files;
            if a file doesn't exist, skip reading and use default parameters.
        """
        print("Load models..", flush=True)
        if os.path.exists(self.a2cnet_fname):
            self.a2c_model = torch.load(self.a2cnet_fname)
        else:
            print("A2C network does not exist, and so, not loaded",
                  flush=True)
        if os.path.exists(self.optimizer_fname):
            # The optimizer needs to do two phases to correctly link it
            # to the policy network, and load parameters.
            loaded_optimizer = torch.load(self.optimizer_fname)
            self.optimizer.load_state_dict(loaded_optimizer.state_dict())
        else:
            print("Optimizer  does not exist, and so, not loaded", flush=True)

    def save_model(self):
        """ Save a2c model and optimizer parameters to files. """
        if not self.is_training_mode():
            return
        print("Save models..", flush=True)
        torch.save(self.a2c_model, self.a2cnet_fname)
        torch.save(self.optimizer, self.optimizer_fname)

    def load_best_network(self):
        pass

    def save_best_network(self):
        if self.is_training_mode():
            pass
        pass

    def is_training_mode(self):
        return "training" in self.execution_mode

    def set_training_mode(self):
        print("training mode is enabled")
        self.execution_mode = "training"

    def set_test_mode(self):
        print("test mode is enabled")
        self.execution_mode = "test"

    def start_episode(self):
        """ Start a new episode, and update (or initialize) the current state.
        """
        self.episode += 1
        #self.print_model("started")

    def finalize_episode(self):
        """ Finalize the current episode.
        """
        #self.print_model("finished")
        print("Episode total reward:", self.episode, ", ", self.accumulated_reward)
        if self.is_training_mode():
            self.save_model()
            self.accumulated_reward = 0

    def print_model(self, prefix: str):
        with open("models/" + prefix + ".a2c_network.str", "w") as fp:
            for key, param in self.a2c_model.named_parameters():
                fp.write(key + " = " + str(param))
        with open("models/" + prefix + ".optimizer.str", "w") as fp:
            for key, param in self.optimizer.state_dict().items():
                fp.write(key + " = " + str(param))
