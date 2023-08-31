
"""!
@file execute.py
@brief Provides mechanisms to launch and log synthetic task graphs.
"""

import functools
import threading
from typing import Dict, Tuple, Union, List
from dataclasses import dataclass, field

from .threads import Propagate

from .graphs import *

import os
import tempfile
from enum import Enum
import time
import itertools

from parla import Parla, spawn, TaskSpace, parray
from parla import sleep_gil
from parla import sleep_nogil
from parla.common.array import clone_here
from parla.common.globals import get_current_devices, get_current_stream, cupy, CUPY_ENABLED, get_current_context
from parla.common.parray.from_data import asarray
from parla.cython.device_manager import cpu, gpu
from parla.cython.variants import specialize
from parla import gpu_sleep_nogil
from parla.cython.core import gpu_bsleep_nogil, gpu_bsleep_gil
import numpy as np

from fractions import Fraction

PArray = parray.core.PArray


def make_parrays(data_list):
    parray_list = list()
    for i, data in enumerate(data_list):
        parray_list.append( asarray(data, name="data"+str(i)) )
    return parray_list


def estimate_frequency(n_samples=10, ticks=1900000000):
    import cupy as cp
    stream = cp.cuda.get_current_stream()
    cycles = ticks
    device_id = 0

    print(f"Starting GPU Frequency benchmark.")
    times = np.zeros(n_samples)
    for i in range(n_samples):

        start = time.perf_counter()
        gpu_bsleep_nogil(device_id, int(ticks), stream)
        stream.synchronize()
        end = time.perf_counter()
        print(f"...collected frequency sample {i} ", end-start)

        times[i] = (end-start)

    times = times[2:]
    elapsed = np.mean(times)
    estimated_speed = cycles/np.mean(times)
    median_speed = cycles/np.median(times)

    print("Finished Benchmark.")
    print("Estimated GPU Frequency: Mean: ", estimated_speed,
          ", Median: ", median_speed, flush=True)
    
    return estimated_speed


class GPUInfo():

    # approximate average on frontera RTX
    # cycles_per_second = 1919820866.3481758
    # cycles_per_second = 867404498.3008006
    # cycles_per_second = 47994628114801.04
    cycles_per_second = 1949802881.4819772

    def update_cycles(self, cycles: float =None):

        if cycles is None:
            cycles = estimate_frequency()
        
        self.cycles_per_second = cycles

    def get_cycles(self) -> int:
        return self.cycles_per_second
    
_GPUInfo = GPUInfo()

@specialize
def free_sleep(duration: float, config: RunConfig = None):
    sleep_nogil(duration)

@free_sleep.variant(architecture=gpu)
def free_sleep_gpu(duration: float, config: RunConfig = None):
    """
    Assumes all GPUs on the system are the same.
    """
    context = get_current_context()
    stream = get_current_stream()

    cycles_per_second = _GPUInfo.get_cycles()
    ticks = int(cycles_per_second * duration)
    gpu_bsleep_nogil(context.device.index, ticks, stream)

    if config.inner_sync:
        stream.synchronize()

@specialize
def lock_sleep(duration: float, config: RunConfig = None):
    sleep_gil(duration)

@lock_sleep.variant(architecture=gpu)
def lock_sleep_gpu(duration: float, config: RunConfig = None):
    """
    Assumes all GPUs on the system are the same.
    """
    context = get_current_context()
    stream = get_current_stream()

    cycles_per_second = _GPUInfo.get_cycles()
    ticks = int(cycles_per_second * duration)
    gpu_bsleep_gil(context.device.index, ticks, stream)

    if config.inner_sync:
        stream.synchronize()



def generate_data(data_config: Dict[int, DataInfo], data_scale: float, data_movement_type) -> List[np.ndarray]:
    value = 0
    data_list = []
    # If data does not exist, this loop will not be iterated.
    for data_idx in data_config:

        data_info = data_config[data_idx]

        data_location = data_info.location
        data_size = data_info.size

        if data_location == DeviceType.CPU_DEVICE:
            data = np.zeros([data_size, data_scale],
                            dtype=np.float32) + value + 1
            data_list.append(data)

        elif data_location > DeviceType.ANY_GPU_DEVICE:
            import cupy as cp
            with cp.cuda.Device(data_location - 1) as device:
                data = cp.zeros([data_size, data_scale],
                                dtyp=np.float32) + value + 1
                device.synchronize()
                data_list.append(data)
        else:
            raise NotImplementedError("This device is not supported for data")
        value += 1

        
    return data_list


#TODO(wlr): Rewrite this supporting multiple device placement.
def generate_data(data_config: Dict[int, DataInfo], data_scale: float, data_movement_type) -> List[np.ndarray]:

    if data_movement_type == MovementType.NO_MOVEMENT:
        return None
    
    elif data_movement_type == MovementType.LAZY_MOVEMENT:
        data_list = create_arrays(data_config, data_scale)
            
    if data_movement_type == MovementType.EAGER_MOVEMENT:
        data_list = create_arrays(data_config, data_scale)
        data_list = make_parrays(data_list)
        if len(data_list) > 0:
            assert isinstance(data_list[0], PArray)
    '''
    if len(data_list) > 0:
        print("[validation] Generated data type:", type(data_list[0]))
    '''
    return data_list


def get_kernel_info(info: TaskRuntimeInfo, config: RunConfig = None) -> Tuple[float, float, int]:
        task_time = info.task_time
        gil_fraction = info.gil_fraction
        gil_accesses = info.gil_accesses

        if config is not None:
            if config.task_time is not None:
                task_time = config.task_time
            if config.gil_accesses is not None:
                gil_accesses = config.gil_accesses
            if config.gil_fraction is not None:
                gil_fraction = config.gil_fraction

        kernel_time = task_time / gil_accesses
        free_time = kernel_time * (1 - gil_fraction)
        gil_time = kernel_time * gil_fraction

        return (free_time, gil_time), gil_accesses

def synthetic_kernel(runtime_info: Dict[Device | tuple[Device] , TaskRuntimeInfo | Dict[Device, TaskRuntimeInfo]], config: RunConfig):
    """
    A simple synthetic kernel that simulates a task that takes a given amount of time
    and accesses the GIL a given number of times. The GIL is accessed in a fraction of
    the total time given.
    """

    if config.verbose:
        task_internal_start_t = time.perf_counter()

    context = get_current_context()
    devices = convert_context_to_devices(context)
    details = get_placement_info(devices, runtime_info)

    info = []
    for device in devices:
        if isinstance(details, TaskRuntimeInfo):
            info.append(details)
        else:
            info.append(details[device])
        if info is None:
            raise ValueError(f"TaskRuntimeInfo cannot be None for {device}. Please check the runtime info passed to the task.")
        
    waste_time(info, config)

    if config.verbose:
        task_internal_end_t = time.perf_counter()
        task_internal_duration = task_internal_end_t - task_internal_start_t
        return task_internal_duration

    return None

@specialize
def waste_time(info_list: List[TaskRuntimeInfo], config: RunConfig):
    
    if len(info_list) == 0:
        raise ValueError("No TaskRuntimeInfo provided to busy sleep kernel.")
    
    info = info_list[0]

    (free_time, gil_time), gil_accesses = get_kernel_info(info, config=config)

    if gil_accesses == 0:
        free_sleep(free_time)
        return
    
    else:
        for i in range(gil_accesses):
            free_sleep(free_time)
            lock_sleep(gil_time)

@waste_time.variant(architecture=gpu)
def waste_time_gpu(info_list: List[TaskRuntimeInfo], config: RunConfig):

    context = get_current_context()
    if len(info_list) < len(context.devices):
        raise ValueError("Not enough TaskRuntimeInfo provided to busy sleep kernel. Must be equal to number of devices.")
    
    for device in context.loop():
        info = info_list[device.index]
        (free_time, gil_time), gil_accesses = get_kernel_info(info, config=config)
        if gil_accesses == 0:
            free_sleep(free_time, config=config)
        else:
            for i in range(gil_accesses):
                free_sleep(free_time, config=config)
                lock_sleep(gil_time, config=config)

    if config.outer_sync:
        context.synchronize()
    

def build_parla_device(mapping: Device, runtime_info: TaskRuntimeInfo):
    
    if mapping.architecture == Architecture.CPU:
        arch = cpu
    elif mapping.architecture == Architecture.GPU:
        arch = gpu
    elif mapping.architecture == Architecture.ANY:
        arch = None 

    if arch is None:
        return None
    
    device_constraints = get_placement_info(mapping, runtime_info)

    if device_constraints is None:
        raise ValueError(f"Device constraints cannot be None for {mapping}. Please check the runtime info passed to the task.")
    
    device_memory = device_constraints.memory
    device_fraction = device_constraints.device_fraction

    #Instatiate the Parla device object (may require scheduler to be active)
    device = arch(mapping.device_id)({'memory': device_memory, 'vcus': device_fraction})

    return device

def append_placement(device_set: Device | Tuple[Device], 
                       task_runtime_info: Dict[Device | Tuple[Device], TaskRuntimeInfo | Dict[Device, TaskRuntimeInfo]],
                       placement_list: List):
    """
    Turn configuration objects into a list of Parla placement objects.
    Runtime configuration is stored in a dictionary of dictionaries:
        MultiDevicePlacementSet -> ParlaDevice -> TaskRuntimeInfo
        SingleDevicePlacementSet -> TaskRuntimeInfo
    This assumes Parla 
    """
    
    if isinstance(device_set, Device):
        parla_device, task_runtime_info = build_parla_device(
            device_set, task_runtime_info)
        placement_list.append(parla_device)
    elif isinstance(device_set, tuple):
        multi_device_set = []
        for device in device_set:
            parla_device = build_parla_device(device, task_runtime_info)
            multi_device_set.append(parla_device)
        multi_device_set = tuple(multi_device_set)
        placement_list.append(multi_device_set)
    else:
        raise ValueError(f"Invalid device type: {device_set}")

def build_parla_placement_list(mapping: Device | Tuple[Device] | None, 
                         task_runtime_info: Dict[Device | Tuple[Device], TaskRuntimeInfo| Dict[Device, TaskRuntimeInfo]], 
                         num_gpus: int):
    placement_list = []

    if mapping is None:
        for device_set in task_runtime_info.keys():
            append_placement(device_set, task_runtime_info, placement_list)
    else:
        append_placement(mapping, task_runtime_info, placement_list)

    return placement_list


def parse_task_info(task: TaskInfo, taskspaces: Dict[str, TaskSpace], config: RunConfig, data_list: List):
    """
    Parse a tasks configuration into Parla objects to launch the task.
    """

    # Task ID
    task_idx = task.task_id.task_idx
    taskspace = taskspaces[task.task_id.taskspace]
    task_name = task.task_id 

    # Dependency Info (List of Parla Tasks)
    dependencies = [taskspaces[dep.taskspace][dep.task_idx] for dep in task.task_dependencies]

    # Valid Placement Set
    runtime_info = task.task_runtime
    placement_set = build_parla_placement_list(task.mapping, runtime_info, config.num_gpus)

    #Data information
    data_information = task.data_dependencies

    if config.movement_type == MovementType.EAGER_MOVEMENT:
        read_data_list = data_information.read
        write_data_list = data_information.write
        rw_data_list = data_information.read_write
    else:
        read_data_list = []
        write_data_list = []
        rw_data_list = []

    # Remove duplicated data blocks between in/out and inout
    if len(read_data_list) > 0 and len(rw_data_list) > 0:
        read_data_list = list(
            set(read_data_list).difference(set(rw_data_list)))
    if len(write_data_list) > 0 and len(rw_data_list) > 0:
        write_data_list = list(
            set(write_data_list).difference(set(rw_data_list)))

    # Construct data blocks.
    INOUT = [] if len(rw_data_list) == 0 else [
            data_list[d] for d in rw_data_list
        ]
    
    IN = [] if len(read_data_list) == 0 else [
            data_list[d] for d in read_data_list
        ]
    OUT = [] if len(write_data_list) == 0 else [
            data_list[d] for d in write_data_list
        ]
    
    return task_name, (task_idx, taskspace, dependencies, placement_set), (IN, OUT, INOUT), runtime_info
    
def create_task(task_name, task_info, data_info, runtime_info, config: RunConfig):
    try:
        task_idx, T, dependencies, placement_set = task_info
        IN, OUT, INOUT = data_info
        
        @spawn(T[task_idx], dependencies=dependencies, placement=placement_set, input=IN, out=OUT, inout=INOUT)
        async def task_func():

            if config.verbose:
                print(f"+{task_name} Running", flush=True)

            elapsed = synthetic_kernel(runtime_info, config=config)

            if config.verbose:
                print(f"-{task_name} Finished: {elapsed} seconds", flush=True)

    except Exception as e:
        print(f"Failed creating Task {task_name}: {e}", flush=True)
    finally:
        return

def execute_tasks(taskspaces, tasks: Dict[TaskID, TaskInfo], run_config: RunConfig, data_list=None):

    spawn_start_t = time.perf_counter()

    # Spawn tasks
    for task, details in tasks.items():
        task_name, task_info, data_info, runtime_info = parse_task_info(details, taskspaces, run_config, data_list)
        create_task(task_name, task_info, data_info, runtime_info, run_config)

    spawn_end_t = time.perf_counter()

    return taskspaces


def execute_graph(data_config: Dict[int, DataInfo], tasks: Dict[TaskID, TaskInfo], run_config: RunConfig, timing: List[TimeSample]):

    @spawn(vcus=0)
    async def main_task():

        graph_times = []

        for i in range(run_config.inner_iterations):
            data_list = generate_data(
                data_config, run_config.data_scale, run_config.movement_type)

            # Initialize task spaces
            taskspaces = {}

            for task, details in tasks.items():
                space_name = details.task_id.taskspace
                if space_name not in taskspaces:
                    taskspaces[space_name] = TaskSpace(space_name)

            graph_start_t = time.perf_counter()

            execute_tasks(taskspaces, tasks, run_config, data_list=data_list)

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

    def __init__(self, config: GraphConfig, name: str, graph_path=None):
        self.config = config
        self.graph = None
        self.data_config = None

        self.name = name
        self.graph_function = None

        if isinstance(config, SerialConfig):
            self.graph_function = generate_serial_graph
        elif isinstance(config, IndependentConfig):
            self.graph_function = generate_independent_graph
        elif isinstance(config, ReductionConfig):
            self.graph_function = generate_reduction_graph
        elif isinstance(config, ReductionScatterConfig):
            self.graph_function = generate_reduction_scatter_graph

        if graph_path is not None:
            self.tmpfilepath = graph_path
        else:
            self.tmpfilepath = None

    def __enter__(self):

        self.diro = tempfile.TemporaryDirectory()
        self.dir = self.diro.__enter__()

        if self.tmpfilepath is None:
            self.tmpfilepath = os.path.join(
                self.dir, 'test_'+str(self.name)+'.graph')
        self.tmplogpath = os.path.join(
            self.dir, 'test_'+str(self.name)+'_.blog')

        print("Graph Path:", self.tmpfilepath)
        with open(self.tmpfilepath, 'w') as tmpfile:
            graph = self.graph_function(self.config)
            # print(graph)
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
