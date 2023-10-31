import torch

from typing import Dict, List, Tuple

from ...task import SimulatedTask
from ....types import TaskState, TaskType
from .globals import *

# TODO(hc): make this an rl environment class


class ParlaRLEnvironment:
  # [Features]
  # 1. dependency per device (5, GCN)
  # 2. dependentedependency per state (6 due to dependents, GCN)
  # 3. num. of visible dependents (1, GCN)
  # 4. device per-state load (4 * 5 = 20, FCN)
  gcn_indim = 12
  fcn_indim = 20
  outdim = 4
  device_feature_dim = 4

  def __init__(self):
      self.task_execution_map = dict()

  def create_gcn_task_workload_state(
      self, node_id_offset: int, target_task: SimulatedTask,
      devices: List, taskmap: Dict) -> Tuple[torch.tensor, torch.tensor]:
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
      edge_index = edge_index.to(torch.int64)
      node_features = torch.cat(lst_node_features)
      # Src/dst lists
      assert len(edge_index) == 2
      assert len(node_features) == node_id_offset
      return edge_index, node_features

  def create_task_workload_state(
      self, target_task: SimulatedTask, devices: List,
      taskmap: Dict) -> torch.tensor:
      # The number of dependencies per-state (dim: 4): MAPPED ~ COMPLETED
      # Then, # of the dependencies per-devices
      # (dim: # of devices): 0 ~ # of devices
      current_gcn_state = torch.zeros(self.gcn_indim)
      #print("******** Create GCN states:", target_task)
      # Need to consider the None state due to dependent tasks
      device_state_offset = TaskState.COMPLETED - TaskState.NONE + 1
      for dependency_id in target_task.dependencies:
          dependency = taskmap[dependency_id]
          assigned_device_to_dependency = dependency.assigned_devices
          dependency_state = dependency.state
          dependency_state_offset = (dependency_state - TaskState.NONE)
          # print("  state: ", dependency_state_offset, " = ", current_gcn_state[dependency_state_offset], ", ", dependency_state)
          # The number of the dependencies per state
          current_gcn_state[dependency_state_offset] = \
              current_gcn_state[dependency_state_offset] + 1
          for assigned_device in dependency.assigned_devices:
              # print("  device: ", device_state_offset + assigned_device.device_id, " = ", current_gcn_state[device_state_offset + assigned_device.device_id], ", ", assigned_device.device_id)
              # The number of the dependencies per device
              current_gcn_state[device_state_offset + assigned_device.device_id] = \
                  current_gcn_state[device_state_offset + assigned_device.device_id] + 1
      # The number of the dependent tasks
      current_gcn_state[device_state_offset + len(devices)] = len(target_task.dependents)
      # print(" dependents: ", device_state_offset + len(devices), " = ", current_gcn_state[device_state_offset + len(devices)], ", ", len(target_task.dependents))
      # print("gcn state:", current_gcn_state)
      return current_gcn_state.unsqueeze(0)

  def create_device_load_state(self, target_task: SimulatedTask, devices: List,
                               reservable_tasks: Dict, launchable_tasks: Dict,
                               launched_tasks: Dict) -> torch.tensor:
      """
      Create a state that shows devices' workload states.
      This state will be an input of the fully-connected layer.
      """
      current_state = torch.zeros(self.fcn_indim)
      # print("******** Create states:", target_task)
      # Per-state load per-device
      idx = 0
      for device in devices:
          # print("  ", idx, " = ", len(reservable_tasks[device]), ", ", device)
          # print(launchable_tasks[device])
          # print("  ", idx + 1, " = ", len(launchable_tasks[device][TaskType.COMPUTE]), ", ", device)
          # print("  ", idx + 2, " = ", len(launchable_tasks[device][TaskType.DATA]), ", ", device)
          # print("  ", idx + 3, " = ", len(launched_tasks[device]), ", ", device)
          current_state[idx] = len(reservable_tasks[device])
          current_state[idx + 1] = len(launchable_tasks[device][TaskType.COMPUTE])
          current_state[idx + 2] = len(launchable_tasks[device][TaskType.DATA])
          current_state[idx + 3] = len(launched_tasks[device])
          idx += self.device_feature_dim
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
          target_task, devices, reservable_tasks, launchable_tasks,
          launched_tasks)
      edge_index, current_workload_features = self.create_gcn_task_workload_state(
          0, target_task, devices, taskmap)
      return current_device_load_state, edge_index, current_workload_features

  def create_next_state(self, current_device_load_state, edge_index,
                        current_workload_features, action):
       """
       Create the next state since RL uses a time-difference method to evaluate
       the current model.
       This function increases a chosen device's load by 1, and
       increases the mapped dependency count feature by 1 and decreases the
       spawned dependency count feature by 1.
       """
       next_device_load_state = torch.clone(current_device_load_state)
       next_current_workload_features = torch.clone(current_workload_features)
       num_reservable_task_offset = action * self.device_feature_dim
       # Increase device load
       next_device_load_state[num_reservable_task_offset] = \
           next_device_load_state[num_reservable_task_offset] + 1
       # Increase dependents' states; outgoing edge destinations from the node 0
       # are the dependent tasks.
       # 0th element of the edge_index is a list of the source tasks.
       for i in range(len(edge_index[0])): 
           if edge_index[0][i] == 0:
               # This is not true since dependent task can be on "not-spanwed" state.
               #assert current_workload_features[edge_index[1][i]][TaskState.SPAWNED] != 0
               # One spawned dependency became mapped.
               next_current_workload_features[edge_index[1][i]][TaskState.SPAWNED] = \
                   next_current_workload_features[edge_index[1][i]][TaskState.SPAWNED] - 1
               next_current_workload_features[edge_index[1][i]][TaskState.MAPPED] = \
                   next_current_workload_features[edge_index[1][i]][TaskState.MAPPED] + 1
               # One device selected its device.
               next_current_workload_features[edge_index[1][i]][TaskState.COMPLETED + action] = \
                   next_current_workload_features[edge_index[1][i]][TaskState.COMPLETED + action] + 1
       return next_device_load_state, edge_index, next_current_workload_features


  def calculate_reward(self, task, completion_time):
      if task.name not in self.task_execution_map:
          # print(task.name, "'s completion time:", completion_time)
          self.task_execution_map[task.name] = completion_time
          return torch.tensor([[0]], dtype=torch.float)
      else:
          old_completion_time =  self.task_execution_map[task.name]
          # print(task.name, "'s completion time:", completion_time, " vs ", old_completion_time)
          reward = 1 if old_completion_time > completion_time else 0
          return torch.tensor([[reward]], dtype=torch.float)
