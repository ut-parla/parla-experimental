import pprint
import os
from ast import literal_eval as make_tuple
from fractions import Fraction
from typing import Dict, Tuple
import numpy as np
import subprocess
import re
from typing import NamedTuple, Union
from dataclasses import dataclass, field
import tempfile
import time
from enum import IntEnum

# Synthetic Graphs ENUMS


class DeviceType(IntEnum):
    ANY_DEVICE = -2
    CPU_DEVICE = -1
    ANY_GPU_DEVICE = 0
    GPU_0 = 1
    GPU_1 = 2
    GPU_2 = 3
    GPU_3 = 4


class LogState(IntEnum):
    ADDING_DEPENDENCIES = 0
    ADD_CONSTRAINT = 1
    ASSIGNED_TASK = 2
    START_TASK = 3
    COMPLETED_TASK = 4


class MovementType(IntEnum):
    NO_MOVEMENT = 0
    LAZY_MOVEMENT = 1
    EAGER_MOVEMENT = 2


class DataInitType(IntEnum):
    NO_DATA = 0
    INDEPENDENT_DATA = 1
    OVERLAPPED_DATA = 2


class TaskID(NamedTuple):
    taskspace: str = "T"
    task_idx: Tuple[int] = (0,)
    instance: int = 0


class TaskRuntimeInfo(NamedTuple):
    task_time: float
    device_fraction: Union[float, Fraction]
    gil_accesses: int
    gil_fraction: Union[float, Fraction]
    memory: int


class TaskDataInfo(NamedTuple):
    read: list[int]
    write: list[int]
    read_write: list[int]


class TaskInfo(NamedTuple):
    task_id: TaskID
    task_runtime: Dict[Tuple[int, ...], TaskRuntimeInfo]
    task_dependencies: list[TaskID]
    data_dependencies: TaskDataInfo


class DataInfo(NamedTuple):
    idx: int
    size: int
    location: int


class TaskTime(NamedTuple):
    assigned_t: float
    start_t: float
    end_t: float
    duration: float


class TimeSample(NamedTuple):
    mean: float
    median: float
    std: float
    min: float
    max: float


@dataclass
class TaskConfig:
    task_time: int = 1000
    gil_accesses: int = 1
    gil_fraction: float = 0
    device_fraction: float = 0.25
    memory: int = 0


@dataclass
class TaskConfigs:
    configurations: Dict[Tuple, TaskConfig] = field(default_factory=dict)

    def add(self, device_id, TaskConfig):

        if not isinstance(device_id, tuple):
            device_id = (device_id,)

        device_id = tuple(int(d) for d in device_id)

        # print(device_id)

        self.configurations[device_id] = TaskConfig

    def remove(self, device_id):
        del self.configurations[device_id]


@dataclass
class GraphConfig:
    task_config: TaskConfigs = None
    use_gpus: bool = False
    fixed_placement: bool = False
    data_pattern: int = DataInitType.NO_DATA


@dataclass
class IndependentConfig(GraphConfig):
    task_count: int = 1


@dataclass
class SerialConfig(GraphConfig):
    steps: int = 1
    dependency_count: int = 1
    chains: int = 1


@dataclass
class TreeConfig(GraphConfig):
    levels: int = 1
    branch_factor: int = 2


@dataclass
class RunConfig:
    outer_iterations: int = 1
    inner_iterations: int = 1
    inner_sync: bool = False
    outer_sync: bool = False
    verbose: bool = False
    device_fraction: float = None
    data_scale: float = None
    threads: int = None
    task_time: float = None
    gil_fraction: float = None
    gil_accesses: int = None
    movement_type: int = MovementType.NO_MOVEMENT
    logfile: str = "testing.blog"


task_filter = re.compile(r'InnerTask\{ .*? \}')


def convert_to_dictionary(task_list: list[TaskInfo]) -> dict[TaskID, TaskInfo]:
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


def read_pgraph(filename: str) -> Tuple[dict[int, DataInfo], dict[TaskID, TaskInfo]]:
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


def check_line(line: str) -> int:
    if "Running task" in line:
        return LogState.START_TASK
    elif "Assigned " in line:
        return LogState.ASSIGNED_TASK
    elif "Completed task" in line:
        return LogState.COMPLETED_TASK
    elif "Adding dependencies" in line:
        return LogState.ADDING_DEPENDENCIES
    elif "has constraints" in line:
        return LogState.ADD_CONSTRAINT
    else:
        return None


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


def parse_blog(filename: str = 'parla.blog') -> Tuple[dict[TaskID, TaskTime],  dict[TaskID, list[TaskID]]]:

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

    task_dependencies = {}

    final_instance_map = {}

    for line in output:
        line_type = check_line(line)
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

    return task_times, task_dependencies


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
