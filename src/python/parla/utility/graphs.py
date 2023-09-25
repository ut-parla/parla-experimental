
"""!
@file graphs.py
@brief Provides the core classes for representing and generating synthetic task graphs.
"""


from fractions import Fraction
import numpy as np
import re

from typing import NamedTuple, Union, List, Dict, Tuple, FrozenSet
from dataclasses import dataclass, field
import tempfile
import time
from enum import IntEnum

from collections import defaultdict

from .types import *
from .load import *

# try:
#     from rich import print
#     from rich.traceback import install
#     install(show_locals=False, max_frames=2)
# except ImportError:
#     pass

# Synthetic Graphs ENUMS

graph_generators = []


def register_graph_generator(func):
    """
    Registers a graph generator function to be used for generating synthetic task graphs.
    """
    graph_generators.append(func)
    return func


def shuffle_tasks(tasks: Dict[TaskID, TaskInfo]) -> Dict[TaskID, TaskInfo]:
    """
    Shuffles the task graph
    """
    task_list = list(tasks.values())
    np.random.shuffle(task_list)
    return convert_to_dictionary(task_list)


def get_data_placement(idx, config):
    data_config = config.data_config

    if data_config.architecture == Architecture.CPU:
        return Device(Architecture.CPU, 0)
    if data_config.architecture == Architecture.GPU:
        return Device(Architecture.GPU, idx % config.n_devices)


def check_config(config: GraphConfig):
    """
    Raise warnings for invalid configuration specifications.
    """
    if config is None:
        raise ValueError(
            f"Graph Configuration file must be specified: {config}")

    if config.task_config is None:
        raise ValueError(
            f"Task Configuration file must be specified: {config}")

    if config.data_config is None:
        raise ValueError(
            f"Data Configuration file must be specified: {config}")


@register_graph_generator
def make_independent_graph(config: IndependentConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Generate configuration for data initialization
    if data_config.pattern == DataInitType.NO_DATA:
        data_placement = Device(Architecture.CPU, 0)
        data_dict[0] = DataInfo(0, 1, data_placement)
        num_data_blocks = 1
    else:
        if data_config.pattern == DataInitType.INDEPENDENT_DATA:
            num_data_blocks = config.task_count
        elif data_config.pattern == DataInitType.OVERLAPPED_DATA:
            num_data_blocks = data_config.npartitions
        else:
            raise NotImplementedError(
                f"Data pattern {data_config.pattern} not implemented for independent task graph.")

        data_size = data_config.total_width // num_data_blocks

        for i in range(num_data_blocks):
            data_placement = get_data_placement(i, config)
            data_dict[i] = DataInfo(i, data_size, data_placement)

    # Build task graph
    task_placement_info = configurations
    for i in range(config.task_count):

        # Task ID
        task_id = TaskID("T", (i,), 0)

        # Task Dependencies
        task_dependencies = []

        # Task Data Dependencies
        if data_config.pattern == DataInitType.NO_DATA:
            data_dependencies = TaskDataInfo()
        if data_config.pattern == DataInitType.INDEPENDENT_DATA or data_config.pattern == DataInitType.OVERLAPPED_DATA:
            data_dependencies = TaskDataInfo(
                read=[DataAccess(i % num_data_blocks)])

        # Task Mapping
        if config.fixed_placement:
            if config.placement_arch == Architecture.GPU:
                task_mapping = Device(Architecture.GPU, i % config.n_devices)
            elif config.placement_arch == Architecture.CPU:
                task_mapping = Device(Architecture.CPU, 0)
        else:
            task_mapping = None

        task_dict[task_id] = TaskInfo(
            task_id, task_placement_info, task_dependencies, data_dependencies, task_mapping)

    return task_dict, data_dict


@register_graph_generator
def make_serial_graph(config: SerialConfig) -> Tuple[TaskMap, DataMap]:

    check_config(config)
    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Generate configuration for data initialization
    if data_config.pattern == DataInitType.NO_DATA:
        data_placement = Device(Architecture.CPU, 0)
        data_dict[0] = DataInfo(0, 1, data_placement)
        num_data_blocks = 1
    else:
        if data_config.pattern == DataInitType.INDEPENDENT_DATA:
            num_data_blocks = config.steps * config.chains

        elif data_config.pattern == DataInitType.OVERLAPPED_DATA:
            num_data_blocks = config.chains
        else:
            raise NotImplementedError(
                f"Data pattern {data_config.pattern} not implemented for serial task graph.")

        data_size = data_config.total_width // num_data_blocks

        for i in range(num_data_blocks):
            data_placement = get_data_placement(i, config)
            data_dict[i] = DataInfo(i, data_size, data_placement)

    # Build task graph
    task_placement_info = configurations
    for i in range(config.steps):
        for j in range(config.chains):

            # Task ID:
            task_id = TaskID("T", (i, j), 0)

            # Task Dependencies
            dependency_list = []
            dependency_limit = min(i, config.dependency_count)
            for k in range(1, dependency_limit+1):
                assert (i-k >= 0)
                dependency = TaskID("T", (i-k, j), 0)
                dependency_list.append(dependency)

            # Task Data Dependencies
            if data_config.pattern == DataInitType.NO_DATA:
                data_dependencies = TaskDataInfo()
            else:
                if data_config.pattern == DataInitType.INDEPENDENT_DATA:
                    inout_data_index = i * config.chains + j
                elif data_config.pattern == DataInitType.OVERLAPPED_DATA:
                    inout_data_index = j
                data_dependencies = TaskDataInfo(
                    read_write=[DataAccess(inout_data_index)])

            # Task Mapping
            if config.fixed_placement:
                if config.placement_arch == Architecture.GPU:
                    task_mapping = Device(
                        Architecture.GPU, j % config.n_devices)
                elif config.placement_arch == Architecture.CPU:
                    task_mapping = Device(Architecture.CPU, 0)
            else:
                task_mapping = None

            task_dict[task_id] = TaskInfo(
                task_id, task_placement_info, dependency_list, data_dependencies, task_mapping)

    return task_dict, data_dict


def generate_reduction_graph(config: ReductionConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)

    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Generate configuration for data initialization

    # Build Task Graph
    task_placement_info = configurations

    for level in range(config.levels, -1, -1):
        tasks_in_level = config.branch_factor ** level
        subtree_segment = tasks_in_level / config.num_gpus

        for j in range(tasks_in_level):
            # Task ID:
            task_id = TaskID("T", (level, j), 0)

            # Task Dependencies
            dependency_list = []
            if level < config.levels:
                for k in range(config.branch_factor):
                    dependency = TaskID(
                        "T", (level-1, config.branch_factor*j + k), 0)
                    dependency_list.append(dependency)

            # Task Data Dependencies
            if data_config.pattern == DataInitType.NO_DATA:
                data_dependencies = TaskDataInfo([], [], [])
            else:
                if data_config.pattern == DataInitType.INDEPENDENT_DATA:
                    inout_data_index = level * config.branch_factor + j
                elif data_config.pattern == DataInitType.OVERLAPPED_DATA:
                    inout_data_index = j
                data_dependencies = TaskDataInfo([], [], [inout_data_index])

            # Task Mapping
            if config.fixed_placement:
                if config.placement_arch == Architecture.GPU:
                    task_mapping = Device(
                        Architecture.GPU, j // subtree_segment)
                elif config.placement_arch == Architecture.CPU:
                    task_mapping = Device(Architecture.CPU, 0)
            else:
                task_mapping = None

            task_dict[task_id] = TaskInfo(
                task_id, task_placement_info, dependency_list, data_dependencies, task_mapping)


def generate_reduction_graph(config: ReductionConfig) -> str:
    task_config = config.task_config
    num_gpus = config.num_gpus
    configurations = task_config.configurations

    graph = ""

    if task_config is None:
        raise ValueError("Task config must be specified")

    data_config_string = ""
    if config.data_pattern == DataInitType.NO_DATA:
        data_config_string = "{{1 : -1}}\n"
    elif config.data_pattern == DataInitType.INDEPENDENT_DATA:
        raise NotImplementedError("[Reduction] Data patterns not implemented")
    else:
        single_data_block_size = config.total_data_width
        for i in range(config.branch_factor**config.levels):
            if i > 0:
                data_config_string += ", "
            data_config_string += f"{{ {single_data_block_size} : -1}}"
    data_config_string += "\n"
    graph += data_config_string

    post_configuration_string = ""

    # TODO(hc): when this was designed, we considered multidevice placement.
    #           but for now, we only consider a single device placement and so,
    #           follow the old generator's graph generation rule.

    device_id = Architecture.GPU
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
                        read_dependency += ", "
                write_dependency = f"{config.branch_factor**(reverse_level)*j}"
            else:
                read_dependency = " "
                write_dependency = f"{global_idx}"
            if config.fixed_placement:
                # USER_CHOSEN_DEVICE acts as an offset.
                device_id = int(2 + j // segment)
            else:
                assert device_id == Architecture.CPU_DEVICE or device_id == Architecture.ANY_GPU_DEVICE
            pre_configuration_string = f"{{ {device_id} : "
            configuration_string = pre_configuration_string + post_configuration_string
            graph += f"{reverse_level, j} |  {configuration_string} | {dependency_string} | {read_dependency} : : {write_dependency} \n"
            global_idx += 1
        reverse_level += 1
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
    num_bridge_tasks += 1 if (levels % 2 > 0) else 0
    # Calculate the number of bulk tasks in the graph.
    num_bulk_tasks = (num_tasks - num_bridge_tasks)
    # Calculate the number of bulk tasks per level.
    num_levels_for_bulk_tasks = levels // 2 + 1
    num_bulk_tasks_per_level = num_bulk_tasks // num_levels_for_bulk_tasks
    # All the remaining bulk tasks are added to the last level.
    num_bulk_tasks_last_level = (
        num_bulk_tasks % num_levels_for_bulk_tasks) + num_bulk_tasks_per_level
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
        data_config_string = f"{{1 : -1}}"
    elif tgraph_config.data_pattern == DataInitType.OVERLAPPED_DATA:
        # Each bulk task takes an individual (non-overlapped) data block.
        # A bridge task reduces all data blocks from the bulk tasks in the previous level.
        single_data_block_size = tgraph_config.total_data_width
        for d in range(num_bulk_tasks_last_level):
            if d > 0:
                data_config_string += ", "
            data_config_string += f"{{{single_data_block_size} : -1}}"
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
            l_num_bulk_tasks = num_bulk_tasks_per_level if l < (
                levels - 1) else num_bulk_tasks_last_level
        else:
            l_num_bulk_tasks = num_bulk_tasks_per_level if l < levels else num_bulk_tasks_last_level
        if l % 2 > 0:  # Bridge task condition
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
        else:  # Bulk tasks condition
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
