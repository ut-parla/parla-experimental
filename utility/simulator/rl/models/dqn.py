import math
import random
import torch
import torch.nn
import torch.optim as optim

from ..networks.fcn import *
from .replay_memory import *

class DQNAgent:

    def __init__(self, in_dim: int, out_dim: int,
                 eps_start = 0.9, eps_end = 0.05, eps_decay = 1000,
                 batch_size = 516, gamma = 0.999):
        self.policy_network = FCN(in_dim, out_dim)
        self.target_network = FCN(in_dim, out_dim)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(),
                                       lr=0.002)
        self.replay_memory = ReplayMemory(1000)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # RL parameter setup
        self.n_actions = out_dim
        self.steps = 1
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.episode = 0

    def select_device(self, x):
        """ Select a device (action) with a state `x` and `policy_network`.
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps / self.eps_decay)
        self.steps += 1
        if sample > eps_threshold:
            # TODO(hc): Always use this condition if test mode is enabled
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
        # TODO(hc): Check if the current mode is the training mode.

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

    def load_networks(self):
        pass

    def save_networks(self):
        pass

    def load_optimizer(self):
        pass

    def save_optimizer(self):
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

    def finalize_episode(self):
        """ Finalize the current episode.
        """
        pass
