import os
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from typing import Dict, List, Tuple

from ..networks.a2c_gcn_fcn import *
from ...task import SimulatedTask
from ....types import TaskState, TaskType
from .globals import *

class A2CAgent:
    # TODO(hc): if testing mode is enabled, skip the model optimization.

    def __init__(self, gcn_indim: int, in_dim: int, out_dim: int,
                 execution_mode: str = "training", gamma: float = 0.999):
        self.gcn_indim = gcn_indim
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Current state
        self.current_gcn_state = torch.zeros(gcn_indim)
        # Next state
        self.next_gcn_state = torch.zeros(gcn_indim)
        self.next_state = torch.zeros(in_dim - gcn_indim)

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
        self.execution_mode = execution_mode
        # Interval to update the actor network parameter
        self.step_for_optim = 5
        self.gamma = gamma
        self.episode = 0
        self.a2cnet_fname = "a2c_network.pt"
        self.optimizer_fname = "optimizer.pt"
        self.best_a2cnet_fname = "best_a2c_network.pt"
        self.best_optimizer_fname = "best_optimizer.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def load_models(self):
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

    def save_models(self):
        """ Save a2c model and optimizer parameters to files. """
        if not self.is_training_mode():
            return
        print("Save models..", flush=True)
        torch.save(self.a2c_model, self.a2cnet_fname)
        torch.save(self.optimizer, self.optimizer_fname)

    def load_best_networks(self):
        pass

    def save_best_networks(self):
        if self.is_training_mode():
            pass
        pass

    def is_training_mode(self):
        return "training" in self.execution_mode

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
        with open("models/" + prefix + ".a2c_network.str", "w") as fp:
            for key, param in self.a2c_model.named_parameters():
                fp.write(key + " = " + str(param))
        with open("models/" + prefix + ".optimizer.str", "w") as fp:
            for key, param in self.optimizer.state_dict().items():
                fp.write(key + " = " + str(param))

    def create_gcn_task_workload_state(self, node_id_offset, target_task: SimulatedTask, devices: List, taskmap: Dict) -> Tuple[torch.tensor, torch.tensor]:
        """
        Create a state that shows task workload states.
        This function creates states not only of the current target task, but also
        its adjacent (possibly k-hops in the future) tasks, and its edge list.
        This state will be an input of the GCN layer.
        """
        lst_node_features = []
        lst_src_edge_index = []
        lst_dst_edge_index = []
        # Create a state of the current task, and append it to the features list.
        lst_node_features.append(
            self.create_task_workload_state(target_task, devices, taskmap))
        # This function temporarily assigns an index to each task.
        # This should match the index on the node feature list and the edge list.
        node_id_offset += 1
        for dependency_id in target_task.dependencies:
            dependency = taskmap[dependency_id]
            # Add a dependency to the edge list
            lst_src_edge_index.append(node_id_offset)
            # 0th task is the target task
            lst_dst_edge_index.append(0)
            lst_node_features.append(
                self.create_task_workload_state(dependency, devices, taskmap))
            node_id_offset += 1
        for dependent_id in target_task.dependents:
            dependent = taskmap[dependent_id]
            # 0th task is the target task
            lst_src_edge_index.append(0)
            # Add a dependent to the edge list
            lst_dst_edge_index.append(node_id_offset)
            lst_node_features.append(
                self.create_task_workload_state(dependent, devices, taskmap))
            node_id_offset += 1
        edge_index = torch.Tensor([lst_src_edge_index, lst_dst_edge_index])
        edge_index = edge_index.to(torch.int64).to(device=self.device)
        node_features = torch.cat(lst_node_features).to(device=self.device)
        # Src/dst lists
        assert len(edge_index) == 2
        assert len(node_features) == node_id_offset
        return edge_index, node_features

    def create_task_workload_state(self, target_task: SimulatedTask, devices: List, taskmap: Dict) -> torch.tensor:
        # The number of dependencies per-state (dim: 4): MAPPED ~ COMPLETED
        # Then, # of the dependencies per-devices
        # (dim: # of devices): 0 ~ # of devices
        current_gcn_state = torch.zeros(self.gcn_indim)
        print("******** Create GCN states:", target_task)
        # Need to consider the None state due to dependent tasks
        device_state_offset = TaskState.COMPLETED - TaskState.NONE
        for dependency_id in target_task.dependencies:
            dependency = taskmap[dependency_id]
            assigned_device_to_dependency = dependency.assigned_devices
            dependency_state = dependency.state
            dependency_state_offset = (dependency_state - TaskState.NONE)
            print("  state: ", dependency_state_offset, " = ", current_gcn_state[dependency_state_offset], ", ", dependency_state)
            # The number of the dependencies per state
            current_gcn_state[dependency_state_offset] = \
                current_gcn_state[dependency_state_offset] + 1
            for assigned_device in dependency.assigned_devices:
                print("  device: ", device_state_offset + assigned_device.device_id, " = ", current_gcn_state[device_state_offset + assigned_device.device_id], ", ", assigned_device.device_id)
                # The number of the dependencies per device
                current_gcn_state[device_state_offset + assigned_device.device_id] = \
                    current_gcn_state[device_state_offset + assigned_device.device_id] + 1
        # The number of the dependent tasks
        current_gcn_state[device_state_offset + len(devices)] = len(target_task.dependents)
        print(" dependents: ", device_state_offset + len(devices), " = ", current_gcn_state[device_state_offset + len(devices)], ", ", len(target_task.dependents))
        print("gcn state:", current_gcn_state)
        return current_gcn_state.unsqueeze(0)

    def create_device_load_state(self, target_task: SimulatedTask, devices: List, reservable_tasks: Dict,
                                 launchable_tasks: Dict, launched_tasks: Dict) -> torch.tensor:
        """
        Create a state that shows devices' workload states.
        This state will be an input of the fully-connected layer.
        """
        current_state = torch.zeros(self.in_dim - self.gcn_indim)
        print("******** Create states:", target_task)
        # Per-state load per-device
        idx = 0
        for device in devices:
            print("  ", idx, " = ", len(reservable_tasks[device]), ", ", device)
            print(launchable_tasks[device])
            print("  ", idx + 1, " = ", len(launchable_tasks[device][TaskType.COMPUTE]), ", ", device)
            print("  ", idx + 2, " = ", len(launchable_tasks[device][TaskType.DATA]), ", ", device)
            print("  ", idx + 3, " = ", len(launched_tasks[device]), ", ", device)
            current_state[idx] = len(reservable_tasks[device])
            current_state[idx + 1] = len(launchable_tasks[device][TaskType.COMPUTE])
            current_state[idx + 2] = len(launchable_tasks[device][TaskType.DATA])
            current_state[idx + 3] = len(launched_tasks[device])
            idx += 4
        return current_state

    def create_state(self, target_task: SimulatedTask, devices: List, taskmap: Dict,
                     reservable_tasks: Dict, launchable_tasks: Dict,
                     launched_tasks: Dict):
        current_device_load_state = self.create_device_load_state(
            target_task, devices, reservable_tasks, launchable_tasks, launched_tasks)
        edge_index, node_features = self.create_gcn_task_workload_state(
            0, target_task, devices, taskmap)
        print("Edge index:", edge_index)
        print("Node features:", node_features)
        return current_device_load_state, edge_index, node_features

