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

# Assume that a system has 4 gpus.
num_gpus = 4


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
    USER_CHOSEN_DEVICE = 5


class LogState(IntEnum):
    """
    Specifies the meaning of a log line. Used for parsing the log file.
    """
    ADDING_DEPENDENCIES = 0
    ADD_CONSTRAINT = 1
    ASSIGNED_TASK = 2
    START_TASK = 3
    RUNAHEAD_TASK = 4
    NOTIFY_DEPENDENTS = 6
    COMPLETED_TASK = 6
    UNKNOWN = 7


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

        self.configurations[device_id] = TaskConfig

    def remove(self, device_id):
        del self.configurations[device_id]


@dataclass
class GraphConfig:
    """
    Configures information about generating the synthetic task graph.
    """
    task_config: TaskConfigs = None
    fixed_placement: bool = False
    data_pattern: int = DataInitType.NO_DATA
    total_data_width: int = 2**23
    data_partitions: int = 1
    num_gpus: int = 4


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
class ReductionConfig(GraphConfig):
    """
    Used to configure the generation of a reduction synthetic task graph.
    """
    levels: int = 8  # Number of levels in the tree
    branch_factor: int = 2  # Number of children per node


@dataclass
class ReductionScatterConfig(GraphConfig):
    """
    Used to configure the generation of a reduction-scatter task graph.
    """
    # The total number of tasks.
    # The number of tasks for each level is calculated based on this.
    # e.g., 1000 total tasks and 4 levels, then about 333 tasks exist for each level
    #       with 2 bridge tasks.
    task_count: int = 1
    levels: int = 4 # Number of levels in the tree


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
    do_check: bool = False  # If this is true, validate configuration/execution
    num_gpus: int = 4  # TODO(hc): it is duplicated with GrpahConfig.
    exec_mode: str = "parla" # "test" for RL test, "training" for RL training,
                             # "parla" for parla, "random" for random.


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
        data_info = data_info.split('|')
        idx = 0
        for data in data_info:
            info = data.strip().strip("()").strip().split(",")
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
    elif "Runahead task" in line:
        return LogState.RUNAHEAD_TASK
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

        # If ".dm." is in the task name, ignore it since
        # this is a data movement task.
        # TODO(hc): we may need to verify data movemnt task too.
        if ".dm." in properties['name']:
            continue

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
    task_runahead_times = {}
    task_notify_times = {}
    task_end_times = {}
    task_assigned_times = {}

    task_start_order = []
    task_end_order = []
    task_runahead_order = []

    task_times = {}
    task_states = defaultdict(list)

    task_dependencies = {}

    final_instance_map = {}

    for line in output:
        line_type = check_log_line(line)
        if line_type == LogState.START_TASK:
            start_time = get_time(line)
            task_properties = get_task_properties(line)
            if len(task_properties) == 0:
                # If the length of the task properties is 0,
                # it implies that this task is data movement task.
                # Ignore it.
                continue
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

        elif line_type == LogState.RUNAHEAD_TASK:
            runahead_time = get_time(line)
            task_properties = get_task_properties(line)
            if len(task_properties) == 0:
                # If the length of the task properties is 0,
                # it implies that this task is data movement task.
                # Ignore it.
                continue

            task_properties = task_properties[0]

            current_name = task_properties['name']
            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            task_runahead_times[base_name] = runahead_time
            task_runahead_order.append(base_name)

        elif line_type == LogState.COMPLETED_TASK:
            end_time = get_time(line)
            task_properties = get_task_properties(line)
            if len(task_properties) == 0:
                # If the length of the task properties is 0,
                # it implies that this task is data movement task.
                # Ignore it.
                continue

            task_properties = task_properties[0]

            current_name = task_properties['name']
            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            task_end_times[base_name] = end_time
            task_end_order.append(base_name)

        elif line_type == LogState.NOTIFY_DEPENDENTS:
            notify_time = get_time(line)
            task_properties = get_task_properties(line)
            if len(task_properties) == 0:
                # If the length of the task properties is 0,
                # it implies that this task is data movement task.
                # Ignore it.
                continue

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
            if len(task_properties) == 0:
                # If the length of the task properties is 0,
                # it implies that this task is data movement task.
                # Ignore it.
                continue

            task_properties = task_properties[0]

            current_name = task_properties['name']
            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            task_assigned_times[base_name] = assigned_time

        elif line_type == LogState.ADDING_DEPENDENCIES:
            task_properties = get_task_properties(line)
            if len(task_properties) == 0:
                # If the length of the task properties is 0,
                # it implies that this task is data movement task.
                # Ignore it.
                continue

            current_task = task_properties[0]['name']
            current_dependencies = []

            for d in task_properties[1:]:
                dependency = d['name']
                current_dependencies.append(dependency)

            task_dependencies[current_task] = current_dependencies

    for task in task_end_times:
        assigned_t = task_assigned_times[task]
        start_t = task_start_times[task]
        # end_t = task_end_times[task]
        end_t = task_end_times[task]
        duration = end_t - start_t
        task_times[task] = TaskTime(assigned_t, start_t, end_t, duration)

    return task_times, task_dependencies, task_states


def generate_serial_graph(config: SerialConfig) -> str:
    task_config = config.task_config
    configurations = task_config.configurations

    graph = ""

    data_config_string = ""
    if config.data_pattern == DataInitType.NO_DATA:
        data_config_string = f"{1 , -1}"
    elif config.data_pattern == DataInitType.OVERLAPPED_DATA:
        config.data_partitions = 1
        single_data_block_size = (
            config.total_data_width // config.data_partitions)
        for i in range(config.data_partitions):
            data_config_string += f"{single_data_block_size , -1}"
            if i+1 < config.data_partitions:
                data_config_string += f" | "
    elif config.data_pattern == DataInitType.INDEPENDENT_DATA:
        raise NotImplementedError("[Serial] Data patterns not implemented")
    else:
        raise ValueError(
            f"[Serial] Not supported data configuration: {config.data_pattern}")
    data_config_string += "\n"

    if task_config is None:
        raise ValueError("Task config must be specified")

    configuration_string = ""
    for device_id, task_config in configurations.items():
        last_flag = 1 if device_id == list(
            configurations.keys())[-1] else 0
        if config.fixed_placement:
            device_id = DeviceType.GPU_0
        # Othrewise, expect any cpu or any gpu.
        configuration_string += f"{{ {device_id} : {task_config.task_time}, {task_config.device_fraction}, {task_config.gil_accesses}, {task_config.gil_fraction}, {task_config.memory} }}"

        if last_flag == 0:
            configuration_string += ", "

    graph += data_config_string
    for i in range(config.steps):  # height
        inout_data_index = i
        if config.data_pattern == DataInitType.OVERLAPPED_DATA:
            inout_data_index = 0
        for j in range(config.chains): # width
            # TODO(hc): for now, do not support chain
            dependency_string = ""
            dependency_limit = min(i, config.dependency_count)
            for k in range(1, dependency_limit+1):
                assert (i-k >= 0)
                dependency_string += f"{i-k, j}"

                if k < dependency_limit:
                    dependency_string += " : "

            # TODO(hc): for now, do not support chain
            graph += f"{i, 0} |  {configuration_string} | {dependency_string} | : : {inout_data_index} \n"

    return graph


def generate_reduction_graph(config: ReductionConfig) -> str:
    task_config = config.task_config
    num_gpus = config.num_gpus
    configurations = task_config.configurations

    graph = ""

    if task_config is None:
        raise ValueError("Task config must be specified")

    data_config_string = ""
    if config.data_pattern == DataInitType.NO_DATA:
        data_config_string = f"{1, -1}"
    elif config.data_pattern == DataInitType.INDEPENDENT_DATA:
        raise NotImplementedError("[Reduction] Data patterns not implemented")
    else:
        single_data_block_size = config.total_data_width
        for i in range(config.branch_factor**config.levels):
            if i > 0:
                data_config_string += " | "
            data_config_string += f"{single_data_block_size, -1}"
    data_config_string += "\n"
    graph += data_config_string

    post_configuration_string = ""

    # TODO(hc): when this was designed, we considered multidevice placement.
    #           but for now, we only consider a single device placement and so,
    #           follow the old generator's graph generation rule.

    device_id = DeviceType.ANY_GPU_DEVICE
    for config_device_id, task_config in configurations.items():
        last_flag = 1 if config_device_id == list(
            configurations.keys())[-1] else 0

        post_configuration_string += f"{task_config.task_time}, {task_config.device_fraction}, {task_config.gil_accesses}, {task_config.gil_fraction}, {task_config.memory} }}"
        # TODO(hc): This should be refined.
        #           If users did not set "fixed", then it should be any cpu or gpu.
        device_id = config_device_id[-1]

        if last_flag == 0:
            post_configuration_string += ", "

    reverse_level = 0
    global_idx = 0
    for i in range(config.levels, -1, -1):
        total_tasks_in_level = config.branch_factor ** i
        segment = total_tasks_in_level / num_gpus
        for j in range(total_tasks_in_level):
            if reverse_level > 0:
                dependency_string = " "
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
            if config.fixed_placement:
                # USER_CHOSEN_DEVICE acts as an offset.
                device_id = int(DeviceType.USER_CHOSEN_DEVICE + j // segment)
            else:
                assert device_id == DeviceType.CPU_DEVICE or device_id == DeviceType.ANY_GPU_DEVICE
            pre_configuration_string = f"{{ {device_id} : "
            configuration_string = pre_configuration_string + post_configuration_string
            graph += f"{reverse_level, j} |  {configuration_string} | {dependency_string} | {read_dependency} : : {write_dependency} \n"
            global_idx += 1
        reverse_level += 1
    return graph


def generate_independent_graph(config: IndependentConfig) -> str:
    task_config = config.task_config
    configurations = task_config.configurations
    num_gpus = config.num_gpus

    graph = ""

    data_config_string = ""
    # TODO(hc): for now, assume that data allocation starts from cpu.
    if config.data_pattern == DataInitType.NO_DATA:
        data_config_string = f"{1 , -1}"
    elif config.data_pattern == DataInitType.INDEPENDENT_DATA:
        single_data_block_size = config.total_data_width
        config.data_partitions = 64
        for i in range(config.data_partitions):
            data_config_string += f"{single_data_block_size , -1}"
            if i+1 < config.data_partitions:
                data_config_string += f" | "
    elif config.data_pattern == DataInitType.OVERLAPPED_DATA:
        raise NotImplementedError(
            "[Independent] Data patterns not implemented")
    else:
        raise ValueError("[Independent] Data patterns not implemented")
    data_config_string += "\n"

    if task_config is None:
        raise ValueError("Task config must be specified")

    graph += data_config_string
    for i in range(config.task_count):
        read_data_block = i % config.data_partitions
        configuration_string = ""
        for device_id, task_config in configurations.items():
            last_flag = 1 if device_id == list(
                configurations.keys())[-1] else 0
            if config.fixed_placement:
                device_id = int(DeviceType.USER_CHOSEN_DEVICE + i % num_gpus)
            configuration_string += f"{{ {device_id} : {task_config.task_time}, {task_config.device_fraction}, {task_config.gil_accesses}, {task_config.gil_fraction}, {task_config.memory} }}"
            if last_flag == 0:
                configuration_string += ", "
        graph += f"{i} |  {configuration_string} | | {read_data_block} : :\n"
    return graph


def generate_reduction_scatter_graph(tgraph_config: ReductionScatterConfig) -> str:
    """
    Generate reduction-scatter graph input file.

    e.g.,
    * * * * * * (bulk tasks) 
    \ | | | | /
         *      (bridge task)
    / | | | | \
    * * * * * *
    ...
    """
    task_config = tgraph_config.task_config
    configurations = task_config.configurations
    num_gpus = tgraph_config.num_gpus
    num_tasks = tgraph_config.task_count
    # Level starts from 1
    levels = tgraph_config.levels
    # Calcualte the number of bridge tasks in the graph.
    num_bridge_tasks = levels // 2
    num_bridge_tasks += 1 if (levels % 2 > 0) else  0
    # Calculate the number of bulk tasks in the graph.
    num_bulk_tasks = (num_tasks - num_bridge_tasks)
    # Calculate the number of bulk tasks per level.
    num_levels_for_bulk_tasks = levels // 2 + 1
    num_bulk_tasks_per_level = num_bulk_tasks // num_levels_for_bulk_tasks
    # All the remaining bulk tasks are added to the last level.
    num_bulk_tasks_last_level = (num_bulk_tasks % num_levels_for_bulk_tasks) + num_bulk_tasks_per_level
    # Calculate the number of tasks per gpu per level. 
    num_bulk_tasks_per_gpu = (num_bulk_tasks_per_level) // num_gpus
    """
    for l in range(levels + 1):
        if l % 2 > 0:
            print(f"Level {l}: -- 1 --", flush=True)
        else:
            if l == levels:
                print(f"Level {l}: {num_bulk_tasks_last_level}, {num_bulk_tasks_per_gpu}", flush=True)
            else:
                print(f"Level {l}: {num_bulk_tasks_per_level}, {num_bulk_tasks_per_gpu}", flush=True)
    """

    graph = ""

    data_config_string = ""
    # TODO(hc): for now, assume that data allocation starts from cpu.
    if tgraph_config.data_pattern == DataInitType.NO_DATA:
        data_config_string = f"{1, -1}"
    elif tgraph_config.data_pattern == DataInitType.OVERLAPPED_DATA:
        # Each bulk task takes an individual (non-overlapped) data block.
        # A bridge task reduces all data blocks from the bulk tasks in the previous level.
        single_data_block_size = tgraph_config.total_data_width
        for d in range(num_bulk_tasks_last_level):
            if d > 0:
                data_config_string += " | "
            data_config_string += f"{single_data_block_size, -1}"
    elif tgraph_config.data_pattern == DataInitType.INDEPENDENT_DATA:
        raise NotImplementedError(
            "[Independent] Data patterns not implemented")
    data_config_string += "\n"
    graph += data_config_string

    # TODO(hc): I don't know how to handle user-fixed placement on multi-device
    #           tasks. Let me design single device task workload.
    assert len(task_config.configurations) == 1

    # Construct task graphs.
    task_id = 0
    bridge_task_dev_id = DeviceType.USER_CHOSEN_DEVICE if tgraph_config.fixed_placement else \
                         DeviceType.ANY_GPU_DEVICE
    last_bridge_task_id_str = ""
    last_bridge_task_id = 0 
    for l in range(levels + 1):
        # If the last level has a bridge task, the previous level should take all remaining bulk
        # tasks.
        if levels % 2 > 0:
            l_num_bulk_tasks = num_bulk_tasks_per_level if l < (levels - 1) else num_bulk_tasks_last_level
        else:
            l_num_bulk_tasks = num_bulk_tasks_per_level if l < levels else num_bulk_tasks_last_level
        if l % 2 > 0: # Bridge task condition
            dependency_block = ""
            inout_data_block = ""
            for d in range(l_num_bulk_tasks):
                inout_data_block += f"{d}"
                if l == 1:
                    dependency_block += f"{d + last_bridge_task_id}"
                else:
                    dependency_block += f"{d + last_bridge_task_id + 1}"
                if d != (l_num_bulk_tasks - 1):
                    inout_data_block += ","
                    dependency_block += " : "
            # TODO(hc): assume a single device task.
            for _, sdevice_task_config in task_config.configurations.items():
                graph += (f"{task_id} | {{ {bridge_task_dev_id} : {sdevice_task_config.task_time}, "
                          f"{sdevice_task_config.device_fraction}, {sdevice_task_config.gil_accesses}, "
                          f"{sdevice_task_config.gil_fraction}, {sdevice_task_config.memory} }}")
            graph += f" | {dependency_block}"
            graph += f" | : : {inout_data_block}\n"
            if tgraph_config.fixed_placement:
                bridge_task_dev_id += 1
                if bridge_task_dev_id == num_gpus + DeviceType.USER_CHOSEN_DEVICE:
                    bridge_task_dev_id = DeviceType.USER_CHOSEN_DEVICE
            last_bridge_task_id_str = f"{task_id}"
            last_bridge_task_id = int(task_id)
            task_id += 1
        else: # Bulk tasks condition
            bulk_task_id_per_gpu = 0
            bulk_task_dev_id = DeviceType.USER_CHOSEN_DEVICE if tgraph_config.fixed_placement else \
                               DeviceType.ANY_GPU_DEVICE
            for bulk_task_id in range(l_num_bulk_tasks):
                inout_data_block = f"{bulk_task_id}"
                # TODO(hc): assume a single device task.
                for _, sdevice_task_config in task_config.configurations.items():
                    graph += (f"{task_id} | {{ {bulk_task_dev_id} : {sdevice_task_config.task_time}, "
                              f"{sdevice_task_config.device_fraction}, {sdevice_task_config.gil_accesses}, "
                              f"{sdevice_task_config.gil_fraction}, {sdevice_task_config.memory} }}")
                graph += f" | {last_bridge_task_id_str}"
                graph += f" | : : {inout_data_block}\n"
                l_num_bulk_tasks_per_gpu = l_num_bulk_tasks // num_gpus
                if tgraph_config.fixed_placement:
                    if l_num_bulk_tasks % num_gpus >= (bulk_task_dev_id - DeviceType.USER_CHOSEN_DEVICE):
                        l_num_bulk_tasks_per_gpu += 1
                    bulk_task_id_per_gpu += 1
                    if bulk_task_id_per_gpu == l_num_bulk_tasks_per_gpu:
                        bulk_task_id_per_gpu = 0
                        bulk_task_dev_id += 1
                        if bulk_task_dev_id == num_gpus + DeviceType.USER_CHOSEN_DEVICE:
                            bulk_task_dev_id = DeviceType.USER_CHOSEN_DEVICE
                task_id += 1
    return graph

__all__ = [DeviceType, LogState, MovementType, DataInitType, TaskID, TaskRuntimeInfo,
           TaskDataInfo, TaskInfo, DataInfo, TaskTime, TimeSample, read_pgraph,
           parse_blog, TaskConfigs, RunConfig, shuffle_tasks,
           generate_independent_graph, generate_serial_graph,
           generate_reduction_scatter_graph]
