import math
import os
import random
import torch
import torch.nn
import torch.optim as optim

from ..networks.fcn import *
from .replay_memory import *

class DQNAgent:

    # TODO(hc): execution mode would be enum, instead of string.
    def __init__(self, in_dim: int, out_dim: int, execution_mode: str = "training",
                 eps_start = 0.9, eps_end = 0.05, eps_decay = 1000,
                 batch_size = 10, gamma = 0.999):
        self.policy_network = FCN(in_dim, out_dim)
        self.target_network = FCN(in_dim, out_dim)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(),
                                       lr=0.002)
        self.replay_memory = ReplayMemory(1000)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.execution_mode = execution_mode
        # RL parameter setup
        self.n_actions = out_dim
        self.steps = 1
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.episode = 0
        # File names for loading & storing models.
        self.policynet_fname = "policy_network.pt"
        self.targetnet_fname = "target_network.pt"
        self.optimizer_fname = "optimizer.pt"
        self.best_policynet_fname = "best_policy_network.pt"
        self.best_targetnet_fname = "best_target_network.pt"
        self.best_optimizer_fname = "best_optimizer.pt"

    def is_training_mode(self):
        return "training" in self.execution_mode

    def select_device(self, x):
        """ Select a device (action) with a state `x` and `policy_network`.
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps / self.eps_decay)
        self.steps += 1
        if (not self.is_training_mode()) or sample > eps_threshold:
            with torch.no_grad():
                out = self.policy_network(x)
                for action in range(self.n_actions):
                    max_action_pair = out.max(0)
                    # TODO(hc): may need to check if max action is within
                    # resource available devices.
                    return max_action_pair[1]
        else:
            # TODO(hc): Check mask; the mask is marked if that device does not
            # have sufficient resources.
            out = torch.tensor(
                  random.choice([d for d in range(self.n_actions)]),
                  dtype=torch.float)

        return out

    def optimize_model(self):
        """ Optimize DQN model.
        """

        if not self.is_training_mode():
            return

        if len(self.replay_memory) < self.batch_size:
            return

        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Perform DQN optimization:
        # (reward + gamma * (Q values from the target network with next states))
        # - Q values from the policy network with the current states

        # Make each next state in transitions as a separete element in a list.
        # Then, `target_network` will produce an output for each next state.
        next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state
                                 if s is not None])
        next_states_qvals = self.target_network(next_states).max(1)[0].detach()
        # States should be [[state1], [state2], ..]
        states = torch.cat([s.unsqueeze(0) for s in batch.state])
        # Actions should be [[action1], [action2], ..]
        actions = torch.cat([b.unsqueeze(0) for b in batch.action]).to(self.device)
        # Rewards should be [reward1, reward2, ..]
        rewards = torch.cat([r for r in batch.reward]).to(self.device)
        # Get Q values of the chosen action from `policy_network`.
        states_qvals = self.policy_network(states).gather(1, actions)
        # This is expectated Q value calculation by using the bellmann equation.
        expected_qvals = self.gamma * next_states_qvals + rewards
        loss = torch.nn.SmoothL1Loss()(states_qvals, expected_qvals.unsqueeze(1))
        self.optimizer.zero_grad()
        # Perform gradient descent.
        loss.backward()
        # Clamp gradients to stablize optimization.
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        # Update the network parameters.
        self.optimizer.step()

    def update_target_network(self):
        pass

    def load_models(self):
        """ Load policy_network, target_network, and optimizer
            parameters from files; if a file doesn't exist, skip reading
            and use default parameters. """
        print("Load models..", flush=True)
        if os.path.exists(self.policynet_fname):
            self.policy_network = torch.load(self.policynet_fname)
        else:
            print("Policy network does not exist, and so, not loaded",
                  flush=True)
        if os.path.exists(self.targetnet_fname):
            self.target_network = torch.load(self.targetnet_fname)
        else:
            print("Target network does not exist, and so, not loaded",
                  flush=True)
        if os.path.exists(self.optimizer_fname):
            # The optimizer needs to do two phases to correctly link it
            # to the policy network, and load parameters.
            loaded_optimizer = torch.load(self.optimizer_fname)
            self.optimizer.load_state_dict(loaded_optimizer.state_dict())
        else:
            print("Optimizer  does not exist, and so, not loaded", flush=True)

    def save_models(self):
        """ Save policy_network, target_network, and optimizer
            parameters to files. """
        if not is_training_mode():
            return
        print("Save models..", flush=True)
        torch.save(self.policy_network, self.policynet_fname)
        torch.save(self.target_network, self.targetnet_fname)
        torch.save(self.optimizer, self.optimizer_fname)

    def load_best_networks(self):
        pass

    def save_best_networks(self):
        if is_training_mode():
            pass

        pass

    def append_transition(self, state: torch.tensor, action: torch.tensor,
                          next_state: torch.tensor, reward: torch.tensor):
        """ Append (S, A, S', R) to the experience replay memory.
        """
        self.replay_memory.push(state, action, next_state, reward)

    def start_episode(self):
        """ Start a new episode, and update (or initialize) the current state.
        """
        self.episode += 1
        self.print_model("started")

    def finalize_episode(self):
        """ Finalize the current episode.
        """
        self.print_model("finished")

    def print_model(self, prefix: str):
        with open("models/" + prefix + ".policy_network.str", "w") as fp:
            for key, param in self.policy_network.named_parameters():
                fp.write(key + " = " + str(param))
        with open("models/" + prefix + ".target_network.str", "w") as fp:
            for key, param in self.target_network.named_parameters():
                fp.write(key + " = " + str(param))
        with open("models/" + prefix + ".optimizer.str", "w") as fp:
            for key, param in self.optimizer.state_dict().items():
                fp.write(key + " = " + str(param))
