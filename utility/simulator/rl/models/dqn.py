import math
import random

from ..networks.fcn import *
from ..networks.dqn_gcn_fcn import *
from .replay_memory import *
from .globals import *
from .env import *
from .model import *


class DQNAgent(RLModel):

    # TODO(hc): execution mode would be enum, instead of string.
    def __init__(self, gcn_indim: int, fcn_indim: int, outdim: int,
                 execution_mode: str = "training",
                 eps_start = 0.9, eps_end = 0.05, eps_decay = 1000,
                 batch_size = 128, gamma = 0.999):
        self.gcn_indim = gcn_indim
        self.indim = fcn_indim + gcn_indim
        self.outdim = outdim
        if True: # Use GCN+FCN layers
            self.policy_network = DQNNetwork(gcn_indim, self.indim, outdim)
            self.target_network = DQNNetwork(gcn_indim, self.indim, outdim)
        else: # Use pure FCN layers
            self.policy_network = FCN(indim, outdim)
            self.target_network = FCN(indim, outdim)
        self.mapping_transition_buffer = dict()
        self.complete_transition_list = list()
        self.optimizer = optim.RMSprop(self.policy_network.parameters(),
                                       lr=0.001)
        self.replay_memory = ReplayMemory(256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.execution_mode = execution_mode
        # RL parameter setup
        self.n_actions = outdim
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

    def select_device(self, task, x, gcn_x = None, gcn_edgeindex = None):
        """ Select a device (action) with a state `x` and `policy_network`.
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps / self.eps_decay)
        self.steps += 1
        if (not self.is_training_mode()) or sample > eps_threshold:
            with torch.no_grad():
                model_input = NetworkInput(x, False, gcn_x, gcn_edgeindex)
                out = self.policy_network(model_input)
                print("Out:", out)
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
                  dtype=torch.int64).to(self.device)

        return out

    def optimize_model(self):
        """ Optimize DQN model.
        """
        if not self.is_training_mode():
            return

        if len(self.replay_memory) < self.batch_size:
            return

        print("Optimization starts\n")

        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        """
        print("batch next states:", batch.next_state)
        print("states:", batch.state)
        print("action:", batch.action)
        print("reward:", batch.reward)
        print("gcn states:", batch.gcn_state)
        print("next gcn state:", batch.next_gcn_state)
        """

        # Perform DQN optimization:
        # (reward + gamma * (Q values from the target network with next states))
        # - Q values from the policy network with the current states

        # Make each next state in transitions as a separete element in a list.
        # Then, `target_network` will produce an output for each next state.
        next_states = torch.cat([s.unsqueeze(0) for s in \
                           batch.next_state if s is not None])
        # States should be [[state1], [state2], ..]
        states = torch.cat([s.unsqueeze(0) for s in batch.state])
        # Actions should be [[action1], [action2], ..]
        actions = torch.cat([a.unsqueeze(0) for a in batch.action])

        # Rewards should be [reward1, reward2, ..]
        rewards = torch.cat([r.squeeze(0) for r in batch.reward]).to(self.device)
        # GCN states should be [[state1], [state2], ..] or [].
        # The latter case implies that either a GCN layer is not used, or none
        # of the subgraph is visible.
        lst_gcn_states = [s for s in batch.gcn_state \
                          if s is not None]
        lst_next_gcn_states = [s for s in \
                               batch.next_gcn_state if s is not None]
        # GCN edge index should be [[src nodes], [dst nodes]] or [].
        # The latter case implies that either a GCN layer is not used, or none
        # of the subgraph is visible.
        lst_gcn_edgeindex = [ei for ei in \
                             batch.gcn_edgeindex if ei is not None]
        lst_next_gcn_edgeindex = [ei for ei in \
                                  batch.next_gcn_edgeindex if ei is not None]
        """
        print("lst next states;", next_states)
        print("lst gcn edgeindex:", lst_gcn_edgeindex)
        print("lst gcn states:", lst_next_gcn_states)
        print("lst next gcn ei:", lst_next_gcn_edgeindex)
        print("actions:", actions)
        """
        next_model_inputs = NetworkInput(next_states, True,
                                         lst_next_gcn_states,
                                         lst_next_gcn_edgeindex)
        next_states_qvals = \
            self.target_network(next_model_inputs).max(1)[0].detach()
        model_inputs = NetworkInput(
            states, True, lst_gcn_states, lst_gcn_edgeindex)
        # Get Q values of the chosen action from `policy_network`.
        states_qvals = self.policy_network(model_inputs).gather(1, actions)
        print("optimize model..")
        print("actions:", actions)
        print("state qvals:", states_qvals)
        print("states:", states)
        print("next states:", next_states)
        print("next states qvals:", next_states_qvals)
        print("rewards:", rewards)
        # This is expectated Q value calculation by using the bellmann equation.
        expected_qvals = self.gamma * next_states_qvals + rewards
        """
        print("gamma:", self.gamma * next_states_qvals)
        """
        print("expected qvals:", expected_qvals.unsqueeze(1))
        loss = torch.nn.SmoothL1Loss()(states_qvals, expected_qvals.unsqueeze(1))
        print("loss:", loss)
        self.optimizer.zero_grad()
        # Perform gradient descent.
        loss.backward()
        # Clamp gradients to stablize optimization.
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        # Update the network parameters.
        self.optimizer.step()

    def update_target_network(self):
        """ In DQN, the target network needs to update its parameters
            by the policy network for each inverval. This function
            performs this. """
        if self.episode != 0 and episode % 100 == 0:
            self.target_network.load_state_dict(
                self.policy_network.state_dict())

    def load_model(self):
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

    def save_model(self):
        """ Save policy_network, target_network, and optimizer
            parameters to files. """
        if not self.is_training_mode():
            return
        print("Save models..", flush=True)
        torch.save(self.policy_network, self.policynet_fname)
        torch.save(self.target_network, self.targetnet_fname)
        torch.save(self.optimizer, self.optimizer_fname)

    def load_best_networks(self):
        pass

    def save_best_networks(self):
        if self.is_training_mode():
            pass

        pass

    def append_transition(self, state: torch.tensor, action: torch.tensor,
                          next_state: torch.tensor, reward: torch.tensor,
                          gcn_state = None, gcn_edgeindex = None,
                          next_gcn_state = None, next_gcn_edgeindex = None):
        """ Append (S, A, S', R) to the experience replay memory.
        """
        self.replay_memory.push(state, action, next_state, reward,
                                gcn_state, gcn_edgeindex,
                                next_gcn_state, next_gcn_edgeindex)

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

    def append_statetransition(self, target_task, curr_deviceload_state,
                               edge_index, curr_workload_state,
                               next_deviceload_state, next_workload_state,
                               action):
        # Reward is decided when a task is launched while all the other states and the action are
        # decided when a task is mapped. So temporarily holds state transition information.
        action = action.to(dtype=torch.int64).unsqueeze(0)
        new_transition = Transition(curr_deviceload_state, action,
                                    next_deviceload_state, None,
                                    curr_workload_state, edge_index,
                                    next_workload_state, edge_index)
        self.mapping_transition_buffer[target_task.name] = new_transition

    def complete_statetransition(self, target_task, reward: torch.tensor):
        # TODO(hc): Check if the current mode is the training mode
        complete_transition = self.mapping_transition_buffer[target_task.name]
        complete_transition = complete_transition._replace(reward = reward)
        #self.complete_transition_list.append((target_task.name, complete_transition))
        self.replay_memory.push_transition(complete_transition)
        del self.mapping_transition_buffer[target_task.name]
        self.optimize_model()
