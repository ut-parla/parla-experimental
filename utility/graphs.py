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
import random
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


def get_mapping(
    config: GraphConfig, task_idx: Tuple[int, ...]
) -> Device | Tuple[Device, ...] | None:
    if config.fixed_placement:
        mapping_lambda = config.mapping
        assert mapping_lambda is not None
        task_mapping = mapping_lambda(task_idx)
    else:
        task_mapping = None
    return task_mapping


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
    random.shuffle(task_list)
    return convert_to_dictionary(task_list)


def check_config(config: GraphConfig):
    """
    Raise warnings for invalid configuration specifications.
    """
    if config is None:
        raise ValueError(f"Graph Configuration file must be specified: {config}")

    if config.task_config is None:
        raise ValueError(f"Task Configuration file must be specified: {config}")

    if config.data_config is None:
        raise ValueError(f"Data Configuration file must be specified: {config}")


def get_data_dependencies(
    task_id: TaskID, data_dict: DataMap, data_config: DataGraphConfig
):
    data_dependencies = data_config.edges(task_id)
    for data_id in data_dependencies.all_ids():
        data_placement = data_config.initial_placement(data_id.idx)
        data_size = data_config.data_size
        data_dict[data_id] = DataInfo(data_id, data_size, data_placement)

    return data_dependencies, data_dict


@register_graph_generator
def make_independent_graph(config: IndependentConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build task graph
    for i in range(config.task_count):
        # Task ID
        task_idx = (i,)
        task_id = TaskID("T", task_idx, 0)

        # Task Placement Info
        task_placement_info = configurations(task_id)

        # Task Dependencies
        task_dependencies = []

        # Task Data Dependencies
        data_dependencies, data_dict = get_data_dependencies(
            task_id, data_dict, data_config
        )

        # Task Mapping
        task_mapping = get_mapping(config, task_idx)

        task_dict[task_id] = TaskInfo(
            task_id,
            task_placement_info,
            task_dependencies,
            data_dependencies,
            task_mapping,
        )

    return task_dict, data_dict


@register_graph_generator
def make_serial_graph(config: SerialConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build task graph
    for i in range(config.steps):
        for j in range(config.chains):
            # Task ID:
            task_idx = (i, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Runtime Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            dependency_limit = min(i, config.dependency_count)
            for k in range(1, dependency_limit + 1):
                assert i - k >= 0
                dependency = TaskID("T", (i - k, j), 0)
                dependency_list.append(dependency)

            # Task Data Dependencies
            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    return task_dict, data_dict


@register_graph_generator
def make_cholesky_graph(config: CholeskyConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    for j in range(config.blocks):
        for k in range(j):
            # Inter-block GEMM (update diagonal block)
            syrk_task_id = TaskID("SYRK", (j, k), 0)
            syrk_placement_info = configurations(syrk_task_id)
            dependency_list = [TaskID("SOLVE", (j, k), 0)] + [
                TaskID("SYRK", (j, l), 0) for l in range(k)
            ]
            data_dependencies, data_dict = get_data_dependencies(
                syrk_task_id, data_dict, data_config
            )
            task_mapping = get_mapping(config, (j, k))
            task_dict[syrk_task_id] = TaskInfo(
                syrk_task_id,
                syrk_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

        # Diagonal block Cholesky
        potrf_task_id = TaskID("POTRF", (j,), 0)
        potrf_placement_info = configurations(potrf_task_id)
        dependency_list = [TaskID("SYRK", (j, l), 0) for l in range(j)]
        data_dependencies, data_dict = get_data_dependencies(
            potrf_task_id, data_dict, data_config
        )
        task_mapping = get_mapping(config, (j,))
        task_dict[potrf_task_id] = TaskInfo(
            potrf_task_id,
            potrf_placement_info,
            dependency_list,
            data_dependencies,
            task_mapping,
        )

        for i in range(j + 1, config.blocks):
            for k in range(j):
                # Inter-block GEMM (update off-diagonal block)
                gemm_task_id = TaskID("GEMM", (i, j, k), 0)
                gemm_placement_info = configurations(gemm_task_id)
                dependency_list = [
                    TaskID("SOLVE", (i, k), 0),
                    TaskID("SOLVE", (j, k), 0),
                ] + [TaskID("GEMM", (i, j, l), 0) for l in range(k)]
                data_dependencies, data_dict = get_data_dependencies(
                    gemm_task_id, data_dict, data_config
                )
                task_mapping = get_mapping(config, (i, j, k))
                task_dict[gemm_task_id] = TaskInfo(
                    gemm_task_id,
                    gemm_placement_info,
                    dependency_list,
                    data_dependencies,
                    task_mapping,
                )

            # Panel solve
            solve_task_id = TaskID("SOLVE", (i, j), 0)
            solve_placement_info = configurations(solve_task_id)
            dependency_list = [TaskID("POTRF", (j,), 0)] + [
                TaskID("GEMM", (i, j, l), 0) for l in range(j)
            ]
            data_dependencies, data_dict = get_data_dependencies(
                solve_task_id, data_dict, data_config
            )
            task_mapping = get_mapping(config, (i, j))
            task_dict[solve_task_id] = TaskInfo(
                solve_task_id,
                solve_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    return task_dict, data_dict


@register_graph_generator
def make_reduction_graph(config: ReductionConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)

    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build Task Graph
    count = 0

    for level in range(config.levels - 1, -1, -1):
        tasks_in_level = config.branch_factor ** (level)
        print("tasks in level", tasks_in_level)
        subtree_segment = tasks_in_level / config.n_devices

        for j in range(tasks_in_level):
            # Task ID:
            task_idx = (level, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if level < config.levels - 1:
                for k in range(config.branch_factor):
                    dependency = TaskID(
                        "T", (level + 1, config.branch_factor * j + k), 0
                    )
                    dependency_list.append(dependency)

            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    return task_dict, data_dict


@register_graph_generator
def make_scatter_reduction_graph(config: ScatterReductionConfig):
    check_config(config)

    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build Task Graph

    # Scatter phase
    for level in range(config.levels + 1):
        tasks_in_level = config.branch_factor ** (level)

        subtree_segment = tasks_in_level // config.n_devices

        for j in range(tasks_in_level):
            # Task ID:
            task_idx = (2 * config.levels - level, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if level > 0:
                dependency = TaskID(
                    "T",
                    (2 * config.levels - level + 1, j // config.branch_factor),
                    0,
                )
                dependency_list.append(dependency)

            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    # Reduction phase
    for level in range(config.levels - 1, -1, -1):
        tasks_in_level = config.branch_factor ** (level)
        print("tasks in level", tasks_in_level)
        subtree_segment = tasks_in_level / config.n_devices

        for j in range(tasks_in_level):
            # Task ID:
            task_idx = (level, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            for k in range(config.branch_factor):
                dependency = TaskID("T", (level + 1, config.branch_factor * j + k), 0)
                dependency_list.append(dependency)

            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    return task_dict, data_dict


@register_graph_generator
def make_stencil_graph(config: StencilConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    from rich import print

    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build task graph

    dimensions = tuple(config.width for _ in range(config.dimensions))
    print("dimensions", dimensions)

    for t in range(config.steps):
        grid_generator = np.ndindex(dimensions)
        for grid_tuple in grid_generator:
            # Task ID:
            task_idx = (t,) + grid_tuple
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if t > 0:
                neighbor_generator = tuple(
                    config.neighbor_distance * 2 + 1 for _ in range(config.dimensions)
                )
                stencil_generator = np.ndindex(neighbor_generator)
                for stencil_tuple in stencil_generator:
                    # Filter to only orthogonal stencil directions (no diagonals)
                    # This is inefficient, but allows easy testing of other stencil types
                    stencil_tuple = np.subtract(stencil_tuple, config.neighbor_distance)
                    if np.count_nonzero(stencil_tuple) == 1:
                        dependency_grid = tuple(np.add(grid_tuple, stencil_tuple))
                        print("task_idx", grid_tuple)
                        print("stencil_tuple", stencil_tuple)
                        print("dependency_grid", dependency_grid)
                        out_of_bounds = any(
                            element < 0 or element >= config.width
                            for element in dependency_grid
                        )
                        if not out_of_bounds:
                            dependency = TaskID("T", (t - 1,) + dependency_grid, 0)
                            dependency_list.append(dependency)

            # Task Data Dependencies
            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )
            print(task_dict[task_id])

    return task_dict, data_dict


@register_graph_generator
def make_map_reduce_graph(config: MapReduceConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    from rich import print

    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build task graph
    for i in range(config.steps):
        for j in range(config.width):
            # Task ID:
            task_idx = (0, i, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            if i > 0:
                dependency_list = [TaskID("T", (1, i - 1), 0)]
            else:
                dependency_list = []

            # Task Data Dependencies
            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

        task_idx = (1, i)
        task_id = TaskID("T", task_idx, 0)

        # Task Placement Info
        task_placement_info = configurations(task_id)

        # Task Dependencies
        dependency_list = []
        for j in range(config.width):
            dependency = TaskID("T", (0, i, j), 0)
            dependency_list.append(dependency)

        # Task Data Dependencies
        data_dependencies, data_dict = get_data_dependencies(
            task_id, data_dict, data_config
        )

        # Task Mapping
        task_mapping = get_mapping(config, task_idx)

        task_dict[task_id] = TaskInfo(
            task_id,
            task_placement_info,
            dependency_list,
            data_dependencies,
            task_mapping,
        )

    return task_dict, data_dict


@register_graph_generator
def make_fully_connected_graph(config: FullyConnectedConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    from rich import print

    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    for i in range(config.steps):
        for j in range(config.width):
            # Task ID:
            task_idx = (i, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if i > 0:
                for k in range(config.width):
                    dependency = TaskID("T", (i - 1, k), 0)
                    dependency_list.append(dependency)

            # Task Data Dependencies
            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    return task_dict, data_dict


@register_graph_generator
def make_butterfly_graph(config: ButterflyConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    from rich import print

    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    assert config.steps <= np.log2(config.width) + 1

    # Build task graph
    for i in range(config.steps + 1):
        for j in range(config.width):
            # Task ID:
            task_idx = (i, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if i > 0:
                dependency = TaskID("T", (i - 1, j), 0)
                dependency_list.append(dependency)

                step = 2 ** (config.steps - i)
                print("step", step)

                left_idx = j - step
                if left_idx >= 0 and left_idx < config.width:
                    dependency = TaskID("T", (i - 1, left_idx), 0)
                    dependency_list.append(dependency)

                right_idx = j + step
                if right_idx >= 0 and right_idx < config.width:
                    dependency = TaskID("T", (i - 1, right_idx), 0)
                    dependency_list.append(dependency)

            # Task Data Dependencies
            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    return task_dict, data_dict


@register_graph_generator
def make_sweep_graph(config: SweepConfig) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    from rich import print

    data_config = config.data_config
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    shape = tuple(config.width for _ in range(config.dimensions))

    for i in range(config.steps):
        grid_generator = np.ndindex(shape)

        for grid_tuple in grid_generator:
            # Task ID:
            task_idx = (i,) + grid_tuple
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)
            print(task_placement_info)

            # Task Dependencies
            dependency_list = []
            for j in range(config.dimensions + 1):
                if task_idx[j] > 0:
                    dependency_grid = list(task_idx)
                    dependency_grid[j] -= 1
                    dependency = TaskID("T", tuple(dependency_grid), 0)
                    dependency_list.append(dependency)

            # Task Data Dependencies
            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    return task_dict, data_dict
