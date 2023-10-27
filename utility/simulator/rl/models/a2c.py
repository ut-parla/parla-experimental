import os
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from typing import Dict, List, Tuple
from collections import namedtuple

from ..networks.a2c_gcn_fcn import *
from ...task import SimulatedTask
from ....types import TaskState, TaskType
from .globals import *


A2CTransition = namedtuple("A2CTransition",
                          ("state", "action", "next_state", "reward",
                           # We may store information for a GCN layer too.
                           # If any GCN layer is not used, these should be `None`.
                           "gcn_state", "gcn_edgeindex", "next_gcn_state",
                           "next_gcn_edgeindex"))

class A2CAgent:
    # TODO(hc): if testing mode is enabled, skip the model optimization.

    def __init__(self, gcn_indim: int, in_dim: int, out_dim: int,
                 execution_mode: str = "training", gamma: float = 0.999):
        self.gcn_indim = gcn_indim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_device_load_features = 4

        # Actor: Policy network that selects an action.
        # Critic: Value network that evaluates an action from the policy network.
        self.a2c_model = A2CNetwork(gcn_indim, in_dim, out_dim)
        self.optimizer = optim.RMSprop(self.a2c_model.parameters(),
                                       lr=0.002)
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
        self.mapping_transition_buffer = dict()
        self.task_mapping_decision = dict()
        self.complete_transition_list = list()

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

    def select_device(self, target_task: SimulatedTask, x: torch.tensor, gcn_x = None, gcn_edgeindex = None):
        with torch.autograd.set_detect_anomaly(True):
            with torch.no_grad():
                # Different from DQN, A2C requires gradient tracking.
                model_input = NetworkInput(x, False, gcn_x, gcn_edgeindex)
                # This gets two values:
                # 1) transition probability of all the actions from the current state
                # 2) state value that evaluates the actor's policy;
                #    if a state value and probability distribution are corresponding,
                #    it is a good model.
                actions, value = self.a2c_model(model_input)
                print("Target task:", target_task)
                print("select device:", target_task)
                print("model input:", model_input)
                print("gcn input:", gcn_x)
                print("gcn edgeindex:", gcn_edgeindex)
                print("actions:", actions)
                print("value:", value)
                action_probabilities = F.softmax(actions, dim=0)
                # Sample an action by using Categorical random distribution
                dist = Categorical(action_probabilities)
                action = dist.sample()
                return action

    def optimize_model(self):
        self.steps += 1
        if self.steps == self.step_for_optim:
            with torch.autograd.set_detect_anomaly(True):
                assert len(self.complete_transition_list) == self.step_for_optim
                _, transition = self.complete_transition_list[-1]
                next_x = transition.next_state
                next_gcn_x = transition.next_gcn_state
                next_gcn_edgeindex = transition.next_gcn_edgeindex

                lst_log_probs = []
                lst_values = []
                lst_rewards = []
                entropy_sum = 0
                for target_task, transition_info in self.complete_transition_list:
                    model_input = NetworkInput(
                        transition_info.state, False, transition_info.gcn_state,
                        transition_info.gcn_edgeindex)
                    actions, value = self.a2c_model(model_input)
                    print("optimization task:", target_task)
                    print("states:", transition_info.state)
                    print("gcn states:", transition_info.gcn_state)
                    print("gcn ei:", transition_info.gcn_edgeindex)
                    print("actions:", actions)
                    action_probabilities = F.softmax(actions, dim=0)
                    dist = Categorical(action_probabilities)
                    action = transition_info.action
                    log_prob = dist.log_prob(action)
                    entropy_sum += dist.entropy().mean()
                    lst_log_probs.append(log_prob)
                    lst_values.append(value)
                    lst_rewards.append(transition_info.reward)

                    """
                    previous_task_mapping_decision = self.task_mapping_decision[target_task]
                    print("target task:", target_task)
                    print("action:", action, " previous action:", previous_task_mapping_decision.action)
                    print("state:", transition_info.gcn_state, " previous state:",
                        previous_task_mapping_decision.gcn_state)
                    with torch.no_grad():
                        print("sum:", torch.sum(torch.eq(transition_info.gcn_state,
                                previous_task_mapping_decision.gcn_state)))
                        assert torch.sum(
                            torch.eq(
                              transition_info.state,
                              previous_task_mapping_decision.state)) == \
                            len(transition_info.state)
                        assert torch.sum(
                            torch.eq(
                              transition_info.gcn_state,
                              purevious_task_mapping_decision.gcn_state)) == \
                            len(transition_info.gcn_state) * self.gcn_indim
                        assert action.item() == previous_task_mapping_decision.action.item()
                    """

                assert len(lst_log_probs) == self.step_for_optim
                assert len(lst_values) == self.step_for_optim
                assert len(lst_rewards) == self.step_for_optim

                # To perform TD to optimize the model, get a state value
                # of the expected next state from the critic netowrk
                _, next_value = self.a2c_model(
                    NetworkInput(next_x, False, next_gcn_x, next_gcn_edgeindex))
                cat_rewards = torch.cat(lst_rewards)
                returns = self.compute_returns(next_value, cat_rewards)
                cat_log_probs = torch.cat([lp.unsqueeze(0) for lp in lst_log_probs])
                print("lst_log_probs:", lst_log_probs)
#cat_log_probs = torch.cat(lst_log_probs)
                print("lst values:", lst_values)
#cat_values = torch.cat([v for v in lst_values])
                cat_values = torch.cat(lst_values)
#cat_rewards = torch.cat([r for r in lst_rewards])
                print("lst log probs:", cat_log_probs)
                print("lst values:", cat_values)
                print("lst rewards:", cat_rewards)

                returns = torch.cat(returns).detach() 
                advantage = returns - cat_values
                actor_loss = -(cat_log_probs * advantage.detach())
                actor_loss = actor_loss.mean()
                critic_loss = advantage.pow(2).mean()
                print("actor loss:", actor_loss, ", and critic loss:", critic_loss, " advantage:", advantage)
                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy_sum
                print("loss;", loss)
                self.steps = 0
                self.optimizer.zero_grad()
                print("last values:", cat_values)
                loss.backward()
                """
                for param in self.a2c_model.parameters():
                    print("\t parma:", param)
                    param.grad.data.clamp_(-1, 1)
                """
                self.optimizer.step()
                self.print_model("optimization")
                self.complete_transition_list = list()

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
        device_state_offset = TaskState.COMPLETED - TaskState.NONE + 1
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
            idx += self.num_device_load_features
        return current_state

    def create_state(self, target_task: SimulatedTask, devices: List, taskmap: Dict,
                     reservable_tasks: Dict, launchable_tasks: Dict,
                     launched_tasks: Dict):
        """
        Create teh current state.
        The state consists of two feature types:
        1) Device load state: How many tasks of each state are mapped to each device
        2) Task workload state: How many dependencies are on each state, and how many
                                dependent tasks this task has?
        """
        current_device_load_state = self.create_device_load_state(
            target_task, devices, reservable_tasks, launchable_tasks, launched_tasks)
        edge_index, current_workload_features = self.create_gcn_task_workload_state(
            0, target_task, devices, taskmap)
        print("Edge index:", edge_index)
        print("Node features:", current_workload_features)
        return current_device_load_state, edge_index, current_workload_features

    def create_next_state(self, current_device_load_state, edge_index, current_workload_features, action):
         """
         Create the next state since RL uses a time-difference method to evaluate
         the current model.
         This function increases a chosen device's load by 1, and
         increases the mapped dependency count feature by 1 and decreases the
         spawned dependency count feature by 1.
         """
         next_device_load_state = torch.clone(current_device_load_state)
         next_current_workload_features = torch.clone(current_workload_features)
         num_reservable_task_offset = action * self.num_device_load_features
         # Increase device load
         next_device_load_state[num_reservable_task_offset] = \
             next_device_load_state[num_reservable_task_offset] + 1
         # Increase dependents' states; outgoing edge destinations from the node 0
         # are the dependent tasks.
         # 0th element of the edge_index is a list of the source tasks.
         for i in range(len(edge_index[0])): 
             if edge_index[0][i] == 0:
                 assert current_workload_features[edge_index[1][i]][TaskState.SPAWNED] != 0
                 print("dependent:", edge_index[1][i])
                 # One spawned dependency became mapped.
                 next_current_workload_features[edge_index[1][i]][TaskState.SPAWNED] = \
                     next_current_workload_features[edge_index[1][i]][TaskState.SPAWNED] - 1
                 next_current_workload_features[edge_index[1][i]][TaskState.MAPPED] = \
                     next_current_workload_features[edge_index[1][i]][TaskState.MAPPED] + 1
                 # One device selected its device.
                 next_current_workload_features[edge_index[1][i]][TaskState.COMPLETED + action] = \
                     next_current_workload_features[edge_index[1][i]][TaskState.COMPLETED + action] + 1
         return next_device_load_state, edge_index, next_current_workload_features

    def append_statetransition(self, target_task, curr_deviceload_state, edge_index,
                               curr_workload_state, next_deviceload_state, next_workload_state,
                               action):
        """
        Reward is decided when a task is launched while all the other states and the action are
        decided when a task is mapped. So temporarily holds state transition information.
        """
        new_transition = A2CTransition(curr_deviceload_state, action,
                                       next_deviceload_state, None,
                                       curr_workload_state, edge_index,
                                       next_workload_state, edge_index)
        self.mapping_transition_buffer[target_task.name] = new_transition
        print(target_task, "'s new_transtition is added:", new_transition)


    def complete_statetransition(self, target_task, reward: torch.tensor):
        # TODO(hc): Check if the current mode is the training mode
        complete_transition = self.mapping_transition_buffer[target_task.name]
        print("Complete transition:", complete_transition)
        complete_transition = complete_transition._replace(reward = reward)
        self.complete_transition_list.append((target_task.name, complete_transition))
        del self.mapping_transition_buffer[target_task.name]
        self.optimize_model()


def calculate_reward():
    return torch.tensor([[10]], dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
