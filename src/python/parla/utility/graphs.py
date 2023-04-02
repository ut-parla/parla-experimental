import pprint
import os
from ast import literal_eval as make_tuple
from fractions import Fraction
import numpy as np
import subprocess
import re
from typing import NamedTuple, Union, List, Dict, Tuple
from dataclasses import dataclass, field
import tempfile
import time
from enum import IntEnum

from collections import defaultdict

# Synthetic Graphs ENUMS


class DeviceType(IntEnum):
    """
    Used to specify the valid placement of a device in a synthetic task graph
    """
    ANY_DEVICE = -2
    CPU_DEVICE = -1
    ANY_GPU_DEVICE = 0
    GPU_0 = 1
    GPU_1 = 2
    GPU_2 = 3
    GPU_3 = 4


class LogState(IntEnum):
    """
    Specifies the meaning of a log line. Used for parsing the log file.
    """
    ADDING_DEPENDENCIES = 0
    ADD_CONSTRAINT = 1
    ASSIGNED_TASK = 2
    START_TASK = 3
    NOTIFY_DEPENDENTS = 4
    COMPLETED_TASK = 5
    UNKNOWN = 6


class MovementType(IntEnum):
    """
    Used to specify the type of data movement to be used in a synthetic task graph execution.
    """
    NO_MOVEMENT = 0
    LAZY_MOVEMENT = 1
    EAGER_MOVEMENT = 2


class DataInitType(IntEnum):
    """
    Used to specify the data movement pattern and initialization in a synthetic task graph execution.
    """
    NO_DATA = 0
    INDEPENDENT_DATA = 1
    OVERLAPPED_DATA = 2


class TaskID(NamedTuple):
    """
    The identifier for a task in a synthetic task graph.
    """
    taskspace: str = "T"  # The task space the task belongs to
    task_idx: Tuple[int] = (0,)  # The index of the task in the task space
    # How many times the task has been spawned (continuation number)
    instance: int = 0


class TaskRuntimeInfo(NamedTuple):
    """
    The collection of important runtime information / constraints for a task in a synthetic task graph.
    """
    task_time: float
    device_fraction: Union[float, Fraction]
    gil_accesses: int
    gil_fraction: Union[float, Fraction]
    memory: int


class TaskDataInfo(NamedTuple):
    """
    The data dependencies for a task in a synthetic task graph.
    """
    read: list[int]
    write: list[int]
    read_write: list[int]


class TaskInfo(NamedTuple):
    """
    The collection of important information for a task in a synthetic task graph.
    """
    task_id: TaskID
    task_runtime: Dict[Tuple[int, ...], TaskRuntimeInfo]
    task_dependencies: list[TaskID]
    data_dependencies: TaskDataInfo


class DataInfo(NamedTuple):
    """
    The collection of important information for a data object in a synthetic task graph.
    """
    idx: int
    size: int
    location: int


class TaskTime(NamedTuple):
    """
    The parsed timing information from a task from an execution log.
    """
    assigned_t: float
    start_t: float
    end_t: float
    duration: float


class TimeSample(NamedTuple):
    """
    A collection of timing information.
    """
    mean: float
    median: float
    std: float
    min: float
    max: float
    n: int


@dataclass
class TaskConfig:
    """
    Constraint configuration for a task on a device type in a synthetic task graph.
    """
    task_time: int = 1000
    gil_accesses: int = 1
    gil_fraction: float = 0
    device_fraction: float = 0.25
    memory: int = 0


@dataclass
class TaskConfigs:
    """
    Holds the map of devices to task configurations for a synthetic task graph.
    """
    configurations: Dict[Tuple, TaskConfig] = field(default_factory=dict)

    def add(self, device_id, TaskConfig):

        if not isinstance(device_id, tuple):
            device_id = (device_id,)

        device_id = tuple(int(d) for d in device_id)

        print("device id:", device_id, flush=True)

        self.configurations[device_id] = TaskConfig

    def remove(self, device_id):
        del self.configurations[device_id]


@dataclass
class GraphConfig:
    """
    Configures information about generating the synthetic task graph.
    """
    task_config: TaskConfigs = None
    use_gpus: bool = False
    fixed_placement: bool = False
    data_pattern: int = DataInitType.NO_DATA


@dataclass
class IndependentConfig(GraphConfig):
    """
    Used to configure the generation of an independent synthetic task graph.
    """
    task_count: int = 1


@dataclass
class SerialConfig(GraphConfig):
    """
    Used to configure the generation of a serial synthetic task graph.
    """
    steps: int = 1  # Number of steps in the serial graph chain
    # Number of dependency backlinks per task (used for stress testing)
    dependency_count: int = 1
    chains: int = 1  # Number of chains to generate that can run in parallel


@dataclass
class TreeConfig(GraphConfig):
    """
    Used to configure the generation of a tree synthetic task graph.
    """
    levels: int = 8  # Number of levels in the tree
    branch_factor: int = 2  # Number of children per node


@dataclass
class RunConfig:
    """
    Configuration object for executing a synthetic task graph.
    """
    outer_iterations: int = 1  # Number of times to launch the Parla runtime and execute the task graph
    # Number of times to execute the task graph within the same Parla runtime
    inner_iterations: int = 1
    inner_sync: bool = False  # Whether to synchronize after each kernel launch
    outer_sync: bool = False  # Whether to synchronize at the end of the task
    verbose: bool = False  # Whether to print the task graph to the console
    device_fraction: float = None  # VCUs
    data_scale: float = None  # Scaling factor to increase the size of the data objects
    threads: int = None  # Number of threads to use for the Parla runtime
    # Total time for all tasks (this overrides the time in the graphs)
    task_time: float = None
    # Fraction of time spent in the GIL (this overrides the time in the graphs)
    gil_fraction: float = None
    # Number of kernel launches/GIL accesses per task (this overrides the time in the graphs)
    gil_accesses: int = None
    movement_type: int = MovementType.NO_MOVEMENT  # The data movement pattern to use
    logfile: str = "testing.blog"  # The log file location


task_filter = re.compile(r'InnerTask\{ .*? \}')


def convert_to_dictionary(task_list: List[TaskInfo]) -> Dict[TaskID, TaskInfo]:
    """
    Converts a task list to a task graph dictionary
    """
    task_dict = dict()
    for task in task_list:
        task_dict[task.task_id] = task

    return task_dict


def shuffle_tasks(tasks: Dict[TaskID, TaskInfo]) -> Dict[TaskID, TaskInfo]:
    """
    Shuffles the task graph
    """
    task_list = list(tasks.values())
    np.random.shuffle(task_list)
    return convert_to_dictionary(task_list)


def extract(string: str) -> Union[int, Fraction]:
    """
    Extracts string as decimal or int
    """
    if "." in string:
        return Fraction(string)
    else:
        return int(string)


def read_pgraph(filename: str) -> Tuple[Dict[int, DataInfo], Dict[TaskID, TaskInfo]]:
    """
    Reads a pgraph file and returns:
    1. A list of the nodes in the graph
    2. The initial data configuration
    """

    task_list = []
    data_config = dict()

    with open(filename, 'r') as graph_file:

        lines = graph_file.readlines()

        # Read the initial data configuration
        data_info = lines.pop(0)
        data_info = data_info.split(',')
        # print(data_info)
        idx = 0
        for data in data_info:
            info = data.strip().strip("{}").strip().split(":")
            size = int(info[0].strip())
            location = int(info[1].strip())
            data_config[idx] = DataInfo(idx, size, location)
            idx += 1

        # print("Data Config", data_config)
        # Read the task graph
        for line in lines:

            task = line.split("|")
            # Breaks into [task_id, task_runtime, task_dependencies, data_dependencies]

            # Process task id (can't be empty)
            ids = task[0].strip()
            ids = make_tuple(ids)

            if not isinstance(ids, tuple):
                ids = (ids,)

            if isinstance(ids[0], str) and ids[0].isalpha():
                taskspace = ids[0]
                idx = ids[1]

                if not isinstance(idx, tuple):
                    idx = (idx, )

                task_ids = TaskID(taskspace, idx, 0)
            else:
                taskspace = "T"
                task_ids = TaskID(taskspace, ids, 0)

            # Process task runtime (can't be empty)
            configurations = task[1].strip().split("},")
            task_runtime = dict()
            for config in configurations:
                config = config.strip().strip("{}").strip()
                config = config.split(":")

                targets = config[0].strip().strip("()").strip().split(",")
                targets = [int(target.strip())
                           for target in targets if target.strip() != ""]
                target = tuple(targets)

                details = config[1].strip().split(",")

                details = [extract(detail.strip()) for detail in details]
                details = TaskRuntimeInfo(*details)

                task_runtime[target] = details

            # Process task dependencies (can be empty)
            if len(task) > 2:
                dependencies = task[2].split(":")
                if (len(dependencies) > 0) and (not dependencies[0].isspace()):
                    task_dependencies = []

                    for i in range(len(dependencies)):
                        if not dependencies[i].isspace():
                            ids = dependencies[i].strip()

                            ids = make_tuple(ids)

                            if not isinstance(ids, tuple):
                                ids = (ids,)

                            if isinstance(ids[0], str) and ids[0].isalpha():
                                name, idx = ids[0], ids[1]

                                if not isinstance(idx, tuple):
                                    idx = (idx, )
                                dep_id = TaskID(name, idx, 0)

                            else:
                                dep_id = TaskID(taskspace, ids, 0)

                            task_dependencies.append(dep_id)
                else:
                    task_dependencies = []

            else:
                task_dependencies = []

            task_dependencies = task_dependencies

            # Process data dependencies (can be empty)
            if len(task) > 3:
                # Split into [read, write, read/write]
                types = task[3].split(":")

                check = [not t.isspace() for t in types]

                if any(check):
                    task_data = [[], [], []]

                    for i in range(len(types)):
                        if check[i]:
                            data = types[i].strip().split(",")
                            if not data[0].isspace():
                                task_data[i] = [0 for _ in range(len(data))]

                                for j in range(len(data)):
                                    if not data[j].isspace():
                                        task_data[i][j] = int(data[j])
                else:
                    task_data = [[], [], []]
            else:
                task_data = [[], [], []]

            task_data = TaskDataInfo(*task_data)

            task_tuple = TaskInfo(task_ids, task_runtime,
                                  task_dependencies, task_data)

            task_list.append(task_tuple)

    task_graph = convert_to_dictionary(task_list)

    return data_config, task_graph


def get_time(line: str) -> int:
    logged_time = line.split('>>')[0].strip().strip("\`").strip('[]')
    return int(logged_time)


def check_log_line(line: str) -> int:
    if "Running task" in line:
        return LogState.START_TASK
    elif "Notified dependents" in line:
        return LogState.NOTIFY_DEPENDENTS
    elif "Assigned " in line:
        return LogState.ASSIGNED_TASK
    elif "Completed task" in line:
        return LogState.COMPLETED_TASK
    elif "Adding dependencies" in line:
        return LogState.ADDING_DEPENDENCIES
    elif "has constraints" in line:
        return LogState.ADD_CONSTRAINT
    else:
        return LogState.UNKNOWN


def convert_task_id(task_id: str, instance: int = 0) -> TaskID:
    id = task_id.strip().split('_')
    taskspace = id[0]
    task_idx = tuple([int(i) for i in id[1:]])
    return TaskID(taskspace, task_idx, int(instance))


def get_task_properties(line: str):
    message = line.split('>>')[1].strip()
    tasks = re.findall(task_filter, message)
    tprops = []
    for task in tasks:
        properties = {}
        task = task.strip('InnerTask{').strip('}').strip()
        task_properties = task.split(',')
        for prop in task_properties:
            prop_name, prop_value = prop.strip().split(':')
            properties[prop_name] = prop_value.strip()

        properties['name'] = convert_task_id(
            properties['name'], properties['instance'])

        tprops.append(properties)

    return tprops


def parse_blog(filename: str = 'parla.blog') -> Tuple[Dict[TaskID, TaskTime],  Dict[TaskID, List[TaskID]]]:

    try:
        result = subprocess.run(
            ['bread', '-s', r"-f `[%r] >> %m`", filename], stdout=subprocess.PIPE)

        output = result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        raise Exception(e.output)

    output = output.splitlines()

    task_start_times = {}
    task_end_times = {}
    task_assigned_times = {}

    task_start_order = []
    task_end_order = []

    task_times = {}
    task_states = defaultdict(list)

    task_dependencies = {}

    final_instance_map = {}

    for line in output:
        line_type = check_log_line(line)
        if line_type == LogState.START_TASK:
            start_time = get_time(line)
            task_properties = get_task_properties(line)
            task_properties = task_properties[0]

            task_start_times[task_properties['name']] = start_time
            task_start_order.append(task_properties['name'])

            current_name = task_properties['name']

            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            if base_name in final_instance_map:
                if current_name.instance > final_instance_map[base_name].instance:
                    final_instance_map[base_name] = current_name
            else:
                # if current_name.instance > 0:
                #    raise RuntimeError(
                #        "Instance number is not 0 for first instance of task")
                final_instance_map[base_name] = base_name

        elif line_type == LogState.COMPLETED_TASK:
            end_time = get_time(line)
            task_properties = get_task_properties(line)
            task_properties = task_properties[0]

            current_name = task_properties['name']
            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            task_end_times[base_name] = end_time
            task_end_order.append(base_name)

        elif line_type == LogState.NOTIFY_DEPENDENTS:
            task_properties = get_task_properties(line)
            notifying_task = task_properties[0]
            current_name = notifying_task['name']
            current_state = notifying_task['get_state']
            instance = notifying_task['instance']

            if int(instance) > 0:
                base_name = TaskID(current_name.taskspace,
                                   current_name.task_idx,
                                   0)
                task_states[base_name] += [current_state]

            task_states[current_name] += [current_state]

        elif line_type == LogState.ASSIGNED_TASK:
            assigned_time = get_time(line)
            task_properties = get_task_properties(line)
            task_properties = task_properties[0]

            current_name = task_properties['name']
            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            task_assigned_times[base_name] = assigned_time

        elif line_type == LogState.ADDING_DEPENDENCIES:
            task_properties = get_task_properties(line)
            current_task = task_properties[0]['name']
            current_dependencies = []

            for d in task_properties[1:]:
                dependency = d['name']
                current_dependencies.append(dependency)

            task_dependencies[current_task] = current_dependencies

    for task in task_end_times:
        assigned_t = task_assigned_times[task]
        start_t = task_start_times[task]
        end_t = task_end_times[task]
        duration = end_t - start_t
        task_times[task] = TaskTime(assigned_t, start_t, end_t, duration)

    return task_times, task_dependencies, task_states


def generate_serial_graph(config: SerialConfig) -> str:
    task_config = config.task_config
    configurations = task_config.configurations

    graph = ""

    if config.data_pattern == DataInitType.NO_DATA:
        data_config_string = "{1 : -1}\n"
    else:
        raise NotImplementedError("Data patterns not implemented")

    if task_config is None:
        raise ValueError("Task config must be specified")

    configuration_string = ""
    for device_id, task_config in configurations.items():
        last_flag = 1 if device_id == list(
            configurations.keys())[-1] else 0

        configuration_string += f"{{ {device_id} : {task_config.task_time}, {task_config.device_fraction}, {task_config.gil_accesses}, {task_config.gil_fraction}, {task_config.memory} }}"

        if last_flag == 0:
            configuration_string += ", "

    graph += data_config_string
    for i in range(config.steps):

        for j in range(config.chains):
            dependency_string = ""
            dependency_limit = min(i, config.dependency_count)
            for k in range(1, dependency_limit+1):
                assert (i-k >= 0)
                dependency_string += f"{i-k, j}"

                if k < dependency_limit:
                    dependency_string += " : "

            graph += f"{i, j} |  {configuration_string} | {dependency_string} | \n"

    return graph


def generate_reduction_graph(config: TreeConfig) -> str:
    task_config = config.task_config
    configurations = task_config.configurations

    graph = ""

    if config.data_pattern == DataInitType.NO_DATA:
        data_config_string = "{1 : -1}\n"
    else:
        raise NotImplementedError("Data patterns not implemented")

    if task_config is None:
        raise ValueError("Task config must be specified")

    graph += data_config_string
    post_configuration_string = ""

    # TODO(hc): when this was designed, we considered multidevice placement.
    #           but for now, we only consider a single device placement and so,  
    #           follow the old generator's graph generation rule.

    for device_id, task_config in configurations.items():
        last_flag = 1 if device_id == list(
            configurations.keys())[-1] else 0

#post_configuration_string += f"{{ {device_id} : {task_config.task_time}, {task_config.device_fraction}, {task_config.gil_accesses}, {task_config.gil_fraction}, {task_config.memory} }}"
        post_configuration_string += f"{task_config.task_time}, {task_config.device_fraction}, {task_config.gil_accesses}, {task_config.gil_fraction}, {task_config.memory} }}"

        if last_flag == 0:
            post_configuration_string += ", "

    reverse_level = 0
    global_idx = 0
    for i in range(config.levels, -1, -1):
        total_tasks_in_level = config.branch_factor ** i
        subtree = total_tasks_in_level / 4
        for j in range(total_tasks_in_level):
            if reverse_level > 0:
                dependency_string  = " "
                for k in range(config.branch_factor):
                    dependency_string += f"{reverse_level-1, config.branch_factor*j + k}"
                    if k+1 < config.branch_factor:
                        dependency_string += " : "
            else:
                dependency_string = " "


            if reverse_level > 0:
                l = 0
                read_dependency = " "
                targets = [config.branch_factor**(reverse_level-1)]
                for k in targets:
                    read_dependency += f"{(config.branch_factor**(reverse_level))*j+k}"
                    l += 1
                    if l < len(targets):
                        read_dependency += " , "
                write_dependency = f"{config.branch_factor**(reverse_level)*j}"
            else:
                read_dependency = " "
                write_dependency = f"{global_idx}"
            device_id = int(3 + j // subtree)
            pre_configuration_string = f"{{ {device_id} : "
            configuration_string = pre_configuration_string + post_configuration_string
            graph += f"{reverse_level, j} |  {configuration_string} | {dependency_string} | {read_dependency} : : {write_dependency} \n"
            global_idx += 1
        reverse_level += 1    
    return graph



def generate_independent_graph(config: IndependentConfig) -> str:
    task_config = config.task_config
    configurations = task_config.configurations

    graph = ""

    if config.data_pattern == DataInitType.NO_DATA:
        data_config_string = "{1 : -1}\n"

    if task_config is None:
        raise ValueError("Task config must be specified")

    configuration_string = ""
    for device_id, task_config in configurations.items():
        last_flag = 1 if device_id == list(
            configurations.keys())[-1] else 0

        configuration_string += f"{{ {device_id} : {task_config.task_time}, {task_config.device_fraction}, {task_config.gil_accesses}, {task_config.gil_fraction}, {task_config.memory} }}"

        if last_flag == 0:
            configuration_string += ", "

    graph += data_config_string
    for i in range(config.task_count):
        graph += f"{i} |  {configuration_string} | | \n"

    return graph


__all__ = [DeviceType, LogState, MovementType, DataInitType, TaskID, TaskRuntimeInfo,
           TaskDataInfo, TaskInfo, DataInfo, TaskTime, TimeSample, read_pgraph,
           parse_blog, TaskConfigs, RunConfig, shuffle_tasks,
           generate_independent_graph, generate_serial_graph]
