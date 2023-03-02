import functools
import threading
from typing import Dict, Tuple, Union, List
from dataclasses import dataclass, field

from .graphs import LogState, DeviceType, MovementType, DataInitType, TaskID, TaskRuntimeInfo, TaskDataInfo, TaskInfo, DataInfo, TaskTime, TimeSample
from .graphs import RunConfig, GraphConfig, TaskConfig, TaskConfigs, SerialConfig, IndependentConfig
from .graphs import generate_serial_graph, generate_independent_graph, shuffle_tasks
from .graphs import read_pgraph, parse_blog

import os
import tempfile
from enum import Enum
import time

from parla import Parla, spawn, TaskSpace
from parla import sleep_gil as lock_sleep
from parla import sleep_nogil as free_sleep
import numpy as np

from fractions import Fraction


def generate_data(data_config: Dict[int, DataInfo], data_scale: float) -> Dict[int, np.ndarray]:
    pass


def synthetic_kernel(total_time: int, gil_fraction: Union[Fraction, float], gil_accesses: int, config: RunConfig):
    """
    A simple synthetic kernel that simulates a task that takes a given amount of time
    and accesses the GIL a given number of times. The GIL is accessed in a fraction of
    the total time given.
    """
    if config.verbose:
        task_internal_start_t = time.perf_counter()

    # Simulate task work
    kernel_time = total_time / gil_accesses
    free_time = kernel_time * (1 - gil_fraction)
    gil_time = kernel_time * gil_fraction

    for i in range(gil_accesses):
        free_sleep(free_time)
        lock_sleep(gil_time)

    if config.verbose:
        task_internal_end_t = time.perf_counter()
        task_internal_duration = task_internal_end_t - task_internal_start_t
        return task_internal_duration

    return None


def create_task_no_data(task, taskspaces, config=None, data=None):

    try:
        # Task ID
        task_idx = task.task_id.task_idx
        taskspace = taskspaces[task.task_id.taskspace]

        # Dependency Info
        dependencies = [taskspaces[dep.taskspace][dep.task_idx]
                        for dep in task.task_dependencies]

        # Valid Placement Set
        placement_set = (-1,)  # list(task.task_runtime.keys())

        # TODO: This needs rework with Device support
        runtime_info = task.task_runtime[placement_set]

        # Task Constraints
        device_fraction = runtime_info.device_fraction
        if config.device_fraction is not None:
            device_fraction = config.device_fraction

        # Task Work
        total_time = runtime_info.task_time
        gil_accesses = runtime_info.gil_accesses
        gil_fraction = runtime_info.gil_fraction

        if config.task_time is not None:
            total_time = config.task_time

        if config.gil_accesses is not None:
            gil_accesses = config.gil_accesses

        if config.gil_fraction is not None:
            gil_fraction = config.gil_fraction

        @spawn(taskspace[task_idx], dependencies=dependencies, vcus=device_fraction)
        async def task_func():
            if config.verbose:
                print(f"+{task.task_id} Running", flush=True)

            elapsed = synthetic_kernel(total_time, gil_fraction,
                                       gil_accesses, config=config)

            if config.verbose:
                print(f"-{task.task_id} Finished: {elapsed} seconds", flush=True)

    except Exception as e:
        print(f"Failed creating Task {task.task_id}: {e}", flush=True)
    finally:
        return


def execute_tasks(taskspaces, tasks: Dict[TaskID, TaskInfo], run_config: RunConfig, data=None):

    spawn_start_t = time.perf_counter()

    # Spawn tasks
    for task, details in tasks.items():
        create_task_no_data(details, taskspaces, config=run_config, data=data)

    spawn_end_t = time.perf_counter()

    return taskspaces


def execute_graph(data_config: Dict[int, DataInfo], tasks: Dict[TaskID, TaskInfo], run_config: RunConfig, timing: List[TimeSample]):

    @spawn(vcus=0)
    async def main_task():

        graph_times = []

        for i in range(run_config.inner_iterations):
            data = generate_data(data_config, run_config.data_scale)

            # Initialize task spaces
            taskspaces = {}

            for task, details in tasks.items():
                space_name = details.task_id.taskspace
                if space_name not in taskspaces:
                    taskspaces[space_name] = TaskSpace(space_name)

            graph_start_t = time.perf_counter()

            execute_tasks(taskspaces, tasks, run_config, data=data)

            for taskspace in taskspaces.values():
                await taskspace

            graph_end_t = time.perf_counter()

            graph_elapsed = graph_end_t - graph_start_t
            graph_times.append(graph_elapsed)

        graph_times = np.asarray(graph_times)
        graph_t = TimeSample(np.mean(graph_times), np.median(graph_times), np.std(
            graph_times), np.min(graph_times), np.max(graph_times), len(graph_times))

        timing.append(graph_t)


def run(tasks: Dict[TaskID, TaskInfo], data_config: Dict[int, DataInfo] = None, run_config: RunConfig = None) -> TimeSample:

    if run_config is None:
        run_config = RunConfig(outer_iterations=1, inner_iterations=1,
                               verbose=False, threads=1, data_scale=1)

    timing = []

    for outer in range(run_config.outer_iterations):

        outer_start_t = time.perf_counter()

        with Parla(logfile=run_config.logfile):
            internal_start_t = time.perf_counter()
            execute_graph(data_config, tasks, run_config, timing)
            internal_end_t = time.perf_counter()

        outer_end_t = time.perf_counter()

        parla_total_elapsed = outer_end_t - outer_start_t
        parla_internal_elapsed = internal_end_t - internal_start_t

    return timing[0]


def verify_order(log_times: Dict[TaskID, TaskTime], truth_graph: Dict[TaskID, List[TaskID]]) -> bool:
    """
    Verify that all tasks have run in the correct order in the log graph.
    """

    for task in truth_graph:

        details = truth_graph[task]

        for dependency in details.task_dependencies:
            if log_times[task].start_t < log_times[dependency].end_t:
                print("Task {task} started before dependency {dependency}")
                return False

    return True


def verify_dependencies(log_graph: Dict[TaskID, List[TaskID]], truth_graph: Dict[TaskID, List[TaskID]]):
    """
    Verify that all dependencies in the truth graph have completed execution in the log graph.
    """

    for task in truth_graph:

        details = truth_graph[task]

        for dependency in details.task_dependencies:
            if dependency not in log_graph:
                print(
                    f"Dependency {dependency} of task {task} not in log graph")
                return False

    return True


def verify_complete(log_graph: Dict[TaskID, List[TaskID]], truth_graph: Dict[TaskID, List[TaskID]]) -> bool:
    """
    Verify that all tasks in the truth graph have completed exceution in the log graph.
    """

    for task in truth_graph:
        if task not in log_graph:
            print(f"Task {task} not in log graph")
            return False

    return True


def verify_time(log_times: Dict[TaskID, TaskTime], truth_graph: Dict[TaskID, List[TaskID]], factor: float = 2.0) -> bool:
    """
    Verify that all tasks execute near their expected time.
    """

    for task in truth_graph:
        details = truth_graph[task]

        # TODO: This needs to be fixed for device support
        device_idx = (-1,)  # CPU
        expected_time = details.task_runtime[device_idx].task_time
        observed_time = log_times[task].duration / 1000

        if observed_time > expected_time * factor:
            print(
                f"Task {task} took too long to execute. Expected {expected_time} us, took {observed_time} us")
            return False

    return True


def verify_ntasks(log_times: Dict[TaskID, TaskTime], truth_graph: Dict[TaskID, List[TaskID]]):
    """
    Verify that the number of tasks in the log graph is the same as the number of tasks in the truth graph.
    """

    if len(log_times) != len(truth_graph)+1:
        print(
            f"Number of tasks in log graph ({len(log_times)}) does not match number of tasks in truth graph ({len(truth_graph)})")
        return False

    return True


def verify_states(log_states) -> bool:
    """
    Verify that all tasks have visited all states in a valid order.
    """

    for task in log_states:
        states = log_states[task]
        instance = task.instance

        # if ('SPAWNED' not in states):
        #    print(f"Task {task} did not spawn", flush=True)
        #    return False
        if ('MAPPED' not in states):
            print(f"Task {task} was not mapped.", states, flush=True)
            return False
        if ('RESERVED' not in states):
            print(f"Task {task} was not reserved.", states, flush=True)
            return False
        # if ('RUNNING' not in states):
        #    print(f"Task {task} did not run.", states, flush=True)
        #    return False
        if ('RUNAHEAD' not in states):
            print(f"Task {task} was not runahead", states, flush=True)
            return False

    return True


class Propagate(threading.Thread):
    """
    A wrapper of threading.Thread that propagates exceptions to the main thread. 
    Useful for testing race conditions and timeouts in pytest.
    """

    def __init__(self, target, args=None):
        super().__init__(target=target, args=args)
        self.ex = None
        self.value = None

    def run(self):
        try:
            if self._args is None:
                self.value = self._target()
            else:
                self.value = self._target(*self._args)
        except BaseException as e:
            self.ex = e

    def join(self, timeout=None):
        super().join(timeout)
        if self.ex is not None:
            raise self.ex


def timeout(seconds_before_timeout):
    """
    Decorator that raises an exception if the function takes longer than seconds_before_timeout to execute.
    https://stackoverflow.com/questions/21827874/timeout-a-function-windows
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (
                func.__name__, seconds_before_timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Propagate(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(seconds_before_timeout)
                r = t.value
            except Exception as e:
                print('Unhandled exception in Propagate wrapper', flush=True)
                raise e
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


class GraphContext(object):

    def __init__(self, config: GraphConfig, name: str):
        self.config = config
        self.graph = None
        self.data_config = None

        self.name = name
        self.graph_function = None

        if isinstance(config, SerialConfig):
            self.graph_function = generate_serial_graph
        elif isinstance(config, IndependentConfig):
            self.graph_function = generate_independent_graph

    def __enter__(self):

        self.diro = tempfile.TemporaryDirectory()
        self.dir = self.diro.__enter__()

        self.tmpfilepath = os.path.join(
            self.dir, 'test_'+str(self.name)+'.graph')
        self.tmplogpath = os.path.join(
            self.dir, 'test_'+str(self.name)+'_.blog')

        with open(self.tmpfilepath, 'w') as tmpfile:
            graph = self.graph_function(self.config)
            tmpfile.write(graph)

        self.data_config, self.graph = read_pgraph(self.tmpfilepath)

        return self

    def run(self, run_config: RunConfig, max_time: int = 100):

        @timeout(max_time)
        def run_with_timeout():
            return run(self.graph, self.data_config, run_config)

        return run_with_timeout()

    def __exit__(self, type, value, traceback):
        self.diro.__exit__(type, value, traceback)


__all__ = [run, verify_order, verify_dependencies,
           verify_complete, verify_time, timeout, GraphContext]
