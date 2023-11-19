import copy
import torch

from collections import namedtuple
from functools import partial
from typing import Dict, List, Tuple
from itertools import chain

from ...task import SimulatedTask
from ....types import TaskState, TaskType, Device, Architecture
from .globals import *

# TODO(hc): make this an rl environment class


class ParlaRLBaseEnvironment:
  pass


class ParlaRLEnvironment(ParlaRLBaseEnvironment):
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
          self.task_execution_map[task.name] = completion_time
          return torch.tensor([[0]], dtype=torch.float)
      else:
          old_completion_time =  self.task_execution_map[task.name]
          reward = 1
          if completion_time > 0:
              reward = (old_completion_time - completion_time) / old_completion_time
          if old_completion_time > completion_time:
              self.task_execution_map[task.name] = completion_time
          #print(task.name, "'s completion time:", completion_time, " vs ", old_completion_time, " = ", reward)
          return torch.tensor([[reward]], dtype=torch.float)


class ParlaRLNormalizedEnvironment(ParlaRLBaseEnvironment):
  # [Features]
  # 1. dependency per device (5, GCN)
  # 2. dependentedependency per state (6 due to dependents, GCN)
  # 3. num. of visible dependents (1, GCN)
  # 4. device per-state load (3 * 5 = 15, FCN)
  gcn_indim = 12
  fcn_indim = 15
  outdim = 4
  device_feature_dim = 3

  def __init__(self):
      # Key: task name, value: best task completion time expectation across episodes
      self.task_execution_map = dict()
      # Best execution time measured across episodes
      self.best_execution_time = 999999999999
      # Temporary dictionary to store task execution information.
      # This stores that information for each episode, not across episodes.
      # Depending on the importance of the current episode, mostly decided by the
      # total execution time, this dictionary can either be reflected or not.
      self.candidate_task_execution_map = dict()
      # This tracks how long the previously measured best expected finish time 
      # of a task has not been achieved.
      # If this goal has not been achieved for a long time, this is highly
      # likely due to the result of bad past decisions.
      # (This kind of logs are usually gathered during the first epoch)
      self.num_failed_to_achieved_goals = dict()
      self.remove_threshold = 10

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
      # TODO(hc): need to be normalized
      # The number of dependencies per-state (dim: 4): MAPPED ~ COMPLETED
      # Then, # of the dependencies per-devices
      # (dim: # of devices): 0 ~ # of devices
      current_gcn_state = torch.zeros(self.gcn_indim)
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
      total_tasks = 0
      for device in devices:
          current_state[idx] = len(reservable_tasks[device]) + len(launchable_tasks[device][TaskType.COMPUTE])
          current_state[idx + 1] = len(launchable_tasks[device][TaskType.DATA])
          current_state[idx + 2] = len(launched_tasks[device])
          for i in range(0, self.device_feature_dim):
              total_tasks += current_state[idx + i].item()
          idx += self.device_feature_dim
      # Normalization
      idx = 0
      if (total_tasks > 0):
          for device in devices:
              for i in range(0, self.device_feature_dim):
                  current_state[idx + i] = current_state[idx + i].item() / total_tasks
              idx += self.device_feature_dim
      return current_state

  def create_state(self, target_task: SimulatedTask, devices: List, taskmap: Dict,
                   reservable_tasks: Dict, launchable_tasks: Dict,
                   launched_tasks: Dict):
      """
      Create the current state.
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

  def calculate_reward(self, task, completion_time, is_worth_to_evaluate):
      if task.name not in self.task_execution_map:
          print(task.name, " does not exist on completion time map")
          self.candidate_task_execution_map[task.name] = completion_time
          self.num_failed_to_achieved_goals[task.name] = 0
          return torch.tensor([[0]], dtype=torch.float)
      else:
          old_completion_time =  self.task_execution_map[task.name]
          reward = 1
          if completion_time > 0:
              reward = (old_completion_time - completion_time) / old_completion_time
          if old_completion_time > completion_time:
              self.num_failed_to_achieved_goals[task.name] = 0
              self.candidate_task_execution_map[task.name] = completion_time
          else:
              # The current episode's device selection fails to achieve the goal in
              # the dictionary
              self.num_failed_to_achieved_goals[task.name] += 1
              if self.num_failed_to_achieved_goals[task.name] == self.remove_threshold:
                  # If this goal is hard to achieve across multiple episodes, this might be
                  # because of a somewhat skewed past decisions, and that goal was achieved
                  # by task mapping to other relatively free devices.
                  # In this case, resetting goal might be helpful.
                  self.num_failed_to_achieved_goals[task.name] = 0
                  del self.task_execution_map[task.name]
          print(task.name, "'s completion time:", completion_time, " vs ", old_completion_time, " = reward ", reward)
          if not is_worth_to_evaluate and reward > 0:
              return torch.tensor([[0]], dtype=torch.float)
          return torch.tensor([[reward]], dtype=torch.float)

  def finalize_epoch(self, execution_time):
      execution_time = execution_time.to_float()
      if execution_time < (1.1 * self.best_execution_time):
          print("execution time:", execution_time, " best execution time:", self.best_execution_time)
          for key, value in self.candidate_task_execution_map.items():
              if key not in self.task_execution_map:
                  self.task_execution_map[key] = value
              elif (self.task_execution_map[key] > self.candidate_task_execution_map[key]):
                  self.task_execution_map[key] = value
              else:
                  print("key:", key, ", old value:", self.task_execution_map[key], ", new value:", self.candidate_task_execution_map[key], " fails")
          if execution_time < self.best_execution_time:
              self.best_execution_time = execution_time
      self.candidate_task_execution_map = dict()


class READYSEnvironment(ParlaRLBaseEnvironment):
  # [Features]
  # 1. dependency per device (5, GCN)
  # 2. dependentedependency per state (6 due to dependents, GCN)
  # 3. num. of visible dependents (1, GCN)
  # 4. device per-state load (3 * 5 = 15, FCN)
  gcn_indim = 5
  fcn_indim = 1
  outdim = 4
  device_feature_dim = 3

  def __init__(self):
      self.max_heft = 0

  def get_heft_rank(self, taskmap, task):
      return taskmap[task].heft_rank

  def calculate_heft(self, tasklist, taskmap, devices):
      """
      Calculate HEFT (Heterogeneous Earliest Finish Time) for each task.
      This function assumes that the tasklist is already sorted by a topology.
      The below is the equation:

      HEFT_rank = task_duration + max(HEFT_rank(successors))
      """
      # Iterate from the leaf tasks, and assigns HEFT ranks
      for taskid in reversed(tasklist):
          task = taskmap[taskid]
          task_heft_agent = task.heft_agent
          max_dependent_rank = 0
          duration = 0
          # Get the max HEFT rank among dependent tasks
          for dependent_id in task.dependents:
              dependent = taskmap[dependent_id]
              max_dependent_rank = max(dependent.heft_rank, max_dependent_rank) 
          duration = max([
              task_runtime_info.task_time
              for task_runtime_info in task.get_runtime_info(
                  Device(Architecture.GPU, -1))])
          # Calculate the HEFT rank
          task.heft_rank = duration + max_dependent_rank

      heft_rank_returner = partial(self.get_heft_rank, taskmap)
      # Sort task list by heft rank
      heft_sorted_tasks = sorted(tasklist, key=heft_rank_returner)
      print("heft sorted tasks:", heft_sorted_tasks)

      # After this, each task gets a heft rank (.heft_rank).
      agents = {agent.device_id: [] for agent in devices}
      print("--->", agents)

      HEFTEvent = namedtuple('HEFTEvent', 'task start end')
      #ft = lambda device: 

      for taskid in reversed(heft_sorted_tasks):
          task = taskmap[taskid]
          duration = max([task_runtime_info.task_time
                          for task_runtime_info in task.get_runtime_info(
                              Device(Architecture.GPU, -1))])
          # Dependenices' makespan have already been calculated (should be).
          ready_time = 0
          if len(task.dependencies) > 0:
              ready_time = max([taskmap[dep].heft_makespan for dep in task.dependencies])
          # Check second last task's end time and last task's start time.
          # If that gap fits to the target task's duration time, put that there.
          # If that gap doesn't fit, append the target task.
          earliest_start = -1.0 
          earliest_start_agent = -1
          for agent_id in agents:
              agent = agents[agent_id]
              # TODO(hc): add communication time later
              if len(agent) > 0:
                  candidate_earliest_start = 0
                  any_slack_found = False
                  # Get the end time of the second last task; it tries to
                  # calculate a slack between this and the last task.
                  a = chain([HEFTEvent(None, None, 0)], agent[:-1])
                  for e1, e2 in zip(a, agent):
                      tmp_earliest_start = max(ready_time, e1.end)
                      if e2.start - tmp_earliest_start > duration:
                          # If the last and second lask tasks have enough slack,
                          # schedule that task to the slack.
                          candidate_earliest_start = tmp_earliest_start
                          any_slack_found = True
                          break 
                  if not any_slack_found:
                      # If it failed to find a slack, append this task to the last task.
                      candidate_earliest_start = max(agent[-1].end, ready_time)
              else:
                  # If this agent (device) does not have mapped tasks, the earliest start
                  # time is 0.
                  candidate_earliest_start = 0

              if earliest_start == -1 or earliest_start > candidate_earliest_start:
                  earliest_start_agent = agent_id
                  earliest_start = candidate_earliest_start

          agents[earliest_start_agent].append(
              HEFTEvent(taskid, earliest_start, earliest_start + duration))
          task.heft_makespan = earliest_start + duration
          if task.heft_makespan > self.max_heft:
              self.max_heft = task.heft_makespan

      for key, value in agents.items():
          print("Key:", key)
          for vvalue in value:
              print("span:", taskmap[vvalue.task].heft_makespan, ", ", vvalue)
      

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
      current_gcn_state = torch.zeros(self.gcn_indim)
      # 1) add number of successors
      num_succesors = len(target_task.dependents)

      # 2) add number of predecessors
      num_predecessors = len(target_task.dependencies)

      # 3) task type
      task_type = target_task.task_type

      # 4) ready or not (always 0)
      ready = 1

      # 5) Normalized F
      f_type = task_type
      for dependent_id in target_task.dependents:
          dependent = taskmap[dependent_id]
          dep_task_type = dependent.task_type
          num_dep_deps = len(dependent.dependencies)
          f_type = dep_task_type / float(num_dep_deps)

      current_gcn_state[0] = num_succesors
      current_gcn_state[1] = num_predecessors
      current_gcn_state[2] = task_type
      current_gcn_state[3] = ready
      current_gcn_state[4] = f_type
      print(f"task:{target_task}, 0:{num_succesors}, 1:{num_predecessors}, 2:{task_type}, 3:{ready}, 4:{f_type}")
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
      idx = 0
      total_tasks = 0
      most_idle_device = -1
      most_idle_device_total_tasks = -1
      for device in devices:
          # Ignore CPU
          if device.architecture == Architecture.CPU:
              continue
          dev_num_tasks = len(reservable_tasks[device]) + \
                          len(launchable_tasks[device][TaskType.COMPUTE]) + \
                          len(launchable_tasks[device][TaskType.DATA]) + \
                          len(launched_tasks[device])
          print("device:", device, ", id:", device.device_id, ", ", dev_num_tasks)
          if most_idle_device_total_tasks == -1  or dev_num_tasks < most_idle_device_total_tasks:
              most_idle_device_total_tasks = dev_num_tasks
              most_idle_device = device
      current_state[0] = most_idle_device.device_id
      print(f"task:{target_task}, 0:{most_idle_device}, total: {most_idle_device_total_tasks}")
      return current_state

  def create_state(self, target_task: SimulatedTask, devices: List, taskmap: Dict,
                   reservable_tasks: Dict, launchable_tasks: Dict,
                   launched_tasks: Dict):
      """
      Create the current state.
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

  def calculate_reward(self, task, completion_time, is_worth_to_evaluate):
      print("task:", task, " assigned devices:", task.assigned_devices)
      if task.heft_makespan == 0:
          print("task heft mksp:", task.heft_makespan)
          return torch.tensor([[0]], dtype=torch.float)
      else:
          #reward = (self.max_heft - completion_time) / self.max_heft
          #print("task heft mksp:", self.max_heft, " vs ", completion_time, " = reward:", reward)
          reward = (task.heft_makespan - completion_time) / task.heft_makespan
          print("task heft mksp:", task.heft_makespan, " vs ", completion_time, " = reward:", reward)
          return torch.tensor([[reward]], dtype=torch.float)

  def finalize_epoch(self, execution_time):
      pass
