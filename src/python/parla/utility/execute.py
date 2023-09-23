"""!
@file execute.py
@brief Provides mechanisms to launch and log synthetic task graphs.
"""

import functools
import threading
from typing import Dict, Tuple, Union, List
from dataclasses import dataclass, field

from .threads import Propagate

from .graphs import LogState, DeviceType, MovementType, DataInitType, TaskID, TaskRuntimeInfo, TaskDataInfo, TaskInfo, DataInfo, TaskTime, TimeSample
from .graphs import RunConfig, GraphConfig, TaskConfig, TaskConfigs, SerialConfig, IndependentConfig, ReductionConfig, ReductionScatterConfig
from .graphs import generate_serial_graph, generate_independent_graph, generate_reduction_graph, generate_reduction_scatter_graph, shuffle_tasks
from .graphs import read_pgraph, parse_blog

import os
import tempfile
from enum import Enum
import time
import itertools

from parla import Parla, spawn, TaskSpace, parray
from parla import sleep_gil as lock_sleep
from parla import sleep_nogil as free_sleep
from parla.common.array import clone_here
from parla.common.globals import PyMappingPolicyType
from parla.common.globals import get_current_devices, get_current_stream, get_current_context
from parla.common.parray.from_data import asarray
from parla.cython.device_manager import cpu, gpu
from parla.cython.variants import specialize
from parla.cython.core import gpu_bsleep_nogil
import numpy as np

from fractions import Fraction

PArray = parray.core.PArray


def make_parrays(data_list):
    parray_list = list()
    for i, data in enumerate(data_list):
        parray_list.append(asarray(data, name="data"+str(i)) )
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
    #approximate average on frontera RTX
    #cycles_per_second = 1919820866.3481758
    cycles_per_second = 867404498.3008006
    #cycles_per_second = 875649327.7713356
    #cycles_per_second = 47994628114801.04
#cycles_per_second = 1949802881.4819772
#cycles_per_second = 875649327771.3356
#cycles_per_second = 1002001313000.6014779

    """
    def __init__(self):
        self.cycles_per_second = estimate_frequency()
    """

    def update(self, cycles):
        if cycles is None:
            self.cycles_per_second = cycles

    def get_cycles_per_second(self):
        return self.cycles_per_second
    
_GPUInfo = GPUInfo()

def get_placement_set_from(ps_str_set, num_gpus):
    ps_set = []
    # TODO(hc): This assumes a single device task.
    for ps_str in ps_str_set[0]:
        dev_type = int(ps_str)
        if dev_type == DeviceType.ANY_GPU_DEVICE:
            ps_set.append(gpu)
        elif dev_type == DeviceType.CPU_DEVICE:
            ps_set.append(cpu)
        # TODO(hc): just assume that system has 4 gpus.
        elif dev_type == DeviceType.GPU_0:
            ps_set.append(gpu(0))
        elif dev_type == DeviceType.GPU_1:
            ps_set.append(gpu(1))
        elif dev_type == DeviceType.GPU_2:
            ps_set.append(gpu(2))
        elif dev_type == DeviceType.GPU_3:
            ps_set.append(gpu(3))
        elif dev_type >= DeviceType.USER_CHOSEN_DEVICE:
            gpu_idx = (dev_type - DeviceType.USER_CHOSEN_DEVICE) % num_gpus
            ps_set.append(gpu(gpu_idx))
        else:
            raise ValueError("Does not support this placement:", dev_type)
    return ps_set


def generate_data(data_config: Dict[int, DataInfo], data_scale: float, data_movement_type) -> List[np.ndarray]:
    value = 0
    data_list = []
    # If data does not exist, this loop will not be iterated.
    for data_idx in data_config:
        data_location = data_config[data_idx].location
        data_size = data_config[data_idx].size

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
    if data_movement_type == MovementType.EAGER_MOVEMENT:
        data_list = make_parrays(data_list)
        if len(data_list) > 0:
            assert isinstance(data_list[0], PArray)
    return data_list


@specialize
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

    #print(f"gil accesses: {gil_accesses}, free time: {free_time}, gil time: {gil_time}")

    for i in range(gil_accesses):
        free_sleep(free_time)
        lock_sleep(gil_time)

    if config.verbose:
        task_internal_end_t = time.perf_counter()
        task_internal_duration = task_internal_end_t - task_internal_start_t
        return task_internal_duration

    return None


@synthetic_kernel.variant(architecture=gpu)
def synthetic_kernel_gpu(total_time: int, gil_fraction: Union[Fraction, float], gil_accesses: int, config: RunConfig):
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

    cycles_per_second = _GPUInfo.get_cycles_per_second()
    parla_cuda_stream = get_current_stream()
    ticks = int((total_time/(10**3))*cycles_per_second)

    print(" total time:", total_time, ", cycles_per_second:", cycles_per_second, "device id:", dev_id, " ticks:", ticks, flush=True)

    dev_id = get_current_devices()[0]
    #print(f"gil accesses: {gil_accesses}, free time: {free_time}, gil time: {gil_time}")
    for i in range(gil_accesses):
        #print(dev_id[0]().device_id, parla_cuda_stream.stream, flush=True)
        gpu_bsleep_nogil(dev_id[0]().device_id, int(
            ticks), parla_cuda_stream.stream)
        parla_cuda_stream.stream.synchronize()
        lock_sleep(gil_time)

    if config.verbose:
        task_internal_end_t = time.perf_counter()
        task_internal_duration = task_internal_end_t - task_internal_start_t
        print("Task target duration:", total_time, " Wall clock duration:", task_internal_duration, ", user passed total time:", total_time, ", ticks:", ticks , flush=True)
        return task_internal_duration

    task_internal_end_t = time.perf_counter()
    task_internal_duration = task_internal_end_t - task_internal_start_t
    print("Task target duration:", total_time, " Wall clock duration:", task_internal_duration, ", user passed total time:", total_time, ", ticks:", ticks , flush=True)
    return task_internal_duration


def create_task_no_data(task, taskspaces, config, data_list=None):

    try:
        # Task ID
        task_idx = task.task_id.task_idx
        taskspace = taskspaces[task.task_id.taskspace]

        # Dependency Info
        dependencies = [taskspaces[dep.taskspace][dep.task_idx]
                        for dep in task.task_dependencies]

        # Valid Placement Set
        num_gpus = config.num_gpus
        placement_key = task.task_runtime.keys()
        placement_set_str = list(placement_key)
        placement_set = get_placement_set_from(placement_set_str, num_gpus)

        # TODO: This needs rework with Device support
        # TODO(hc): This assumes that this task is a single task
        #           and does not have multiple placement options.
        runtime_info = task.task_runtime[placement_set_str[0]]

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

        @spawn(taskspace[task_idx], dependencies=dependencies, vcus=device_fraction, placement=placement_set)
        async def task_func():
            if config.verbose:
                print(f"+{task.task_id} Running", flush=True)

            elapsed = synthetic_kernel(total_time, gil_fraction,
                                       gil_accesses, config=config)

            if config.verbose:
                print(f"-{task.task_id} Total time: {total_time} Finished: {elapsed} seconds", flush=True)

    except Exception as e:
        print(f"Failed creating Task {task.task_id}: {e}", flush=True)
    finally:
        return


def create_task_eager_data(task, taskspaces, config=None, data_list=None):

    try:
        # Task ID
        task_idx = task.task_id.task_idx
        taskspace = taskspaces[task.task_id.taskspace]

        # Dependency Info
        dependencies = [taskspaces[dep.taskspace][dep.task_idx]
                        for dep in task.task_dependencies]

        # Valid Placement Set
        num_gpus = config.num_gpus
        placement_key = task.task_runtime.keys()
        placement_set_str = list(placement_key)
        placement_set = get_placement_set_from(placement_set_str, num_gpus)

        # In/out/inout data information
        # read: list, write: list, read_write: list
        data_information = task.data_dependencies
        read_data_list = data_information.read
        write_data_list = data_information.write
        rw_data_list = data_information.read_write

        # Remove duplicated data blocks between in/out and inout
        if len(read_data_list) > 0 and len(rw_data_list) > 0:
            read_data_list = list(
                set(read_data_list).difference(set(rw_data_list)))
        if len(write_data_list) > 0 and len(rw_data_list) > 0:
            write_data_list = list(
                set(write_data_list).difference(set(rw_data_list)))

        """
        print("RW data list:", rw_data_list)
        print("R data list:", read_data_list)
        print("W data list:", write_data_list)
        print("Data list:", data_list)
        """

        # Construct data blocks.
        INOUT = [] if len(rw_data_list) == 0 else [
            (data_list[d], 0) for d in rw_data_list]
        IN = [] if len(read_data_list) == 0 else [(data_list[d], 0)
                                                  for d in read_data_list]
        OUT = [] if len(write_data_list) == 0 else [(data_list[d], 0)
                                                    for d in write_data_list]

        memory_sz = 0
        for inout_parray in INOUT:
            memory_sz += inout_parray[0].nbytes
        for out_parray in OUT:
            memory_sz += out_parray[0].nbytes
        for in_parray in IN:
            memory_sz += in_parray[0].nbytes

        # TODO: This needs rework with Device support
        # TODO(hc): This assumes that this task is a single task
        #           and does not have multiple placement options.
        runtime_info = task.task_runtime[placement_set_str[0]]

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

        """
        print("Eager data in:", IN, " out:", OUT, " inout:", INOUT, flush=True)
        print("task idx:", task_idx, " dependencies:", dependencies, " vcu:", device_fraction,
            " placement:", placement_set)
        # TODO(hc): Add data checking.
        """
        @spawn(taskspace[task_idx], dependencies=dependencies, vcus=device_fraction, placement=[placement_set], input=IN, output=OUT, inout=INOUT)
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


def create_task_lazy_data(task, taskspaces, config=None, data_list=None):

    try:
        # Task ID
        task_idx = task.task_id.task_idx
        taskspace = taskspaces[task.task_id.taskspace]

        # Dependency Info
        dependencies = [taskspaces[dep.taskspace][dep.task_idx]
                        for dep in task.task_dependencies]

        # Valid Placement Set
        num_gpus = config.num_gpus
        placement_key = task.task_runtime.keys()
        placement_set_str = list(placement_key)
        placement_set = get_placement_set_from(placement_set_str, num_gpus)

        # In/out/inout data information
        # read: list, write: list, read_write: list
        data_information = task.data_dependencies
        read_data_list = data_information.read
        write_data_list = data_information.write
        rw_data_list = data_information.read_write

        # Remove duplicated data blocks between in/out and inout
        if len(read_data_list) > 0 and len(rw_data_list) > 0:
            read_data_list = list(
                set(read_data_list).difference(set(rw_data_list)))
        if len(write_data_list) > 0 and len(rw_data_list) > 0:
            write_data_list = list(
                set(write_data_list).difference(set(rw_data_list)))

        """
        print("RW data list:", rw_data_list)
        print("R data list:", read_data_list)
        print("W data list:", write_data_list)
        print("Data list:", data_list)
        """

        # TODO: This needs rework with Device support
        # TODO(hc): This assumes that this task is a single task
        #           and does not have multiple placement options.
        runtime_info = task.task_runtime[placement_set_str[0]]

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

        """
        print("task idx:", task_idx, " dependencies:", dependencies, " vcu:", device_fraction,
              " placement:", placement_set)
        """

        @spawn(taskspace[task_idx], dependencies=dependencies, vcus=device_fraction, placement=[placement_set])
        async def task_func():
            if config.verbose:
                print(f"+{task.task_id} Running", flush=True)

            local_data = dict()

            for d in itertools.chain(read_data_list, rw_data_list):
                data = data_list[d]
                where = -1 if isinstance(data, np.ndarray) else data.device.id
                local_data[d] = clone_here(data)
                old = None
                if config.do_check:
                    old = np.copy(data[0, 1])
                    local_data[d][0, 1] = -old
                if config.verbose:
                    print(
                        f"=Task {task_idx} moved Data[{d}] from Device[{where}]. Block=[{local_data[d][0, 0]}] | Value=[{local_data[d][0, 1]}], <{old}>", flush=True)

            elapsed = synthetic_kernel(total_time, gil_fraction,
                                       gil_accesses, config=config)

            for d in itertools.chain(write_data_list, rw_data_list):
                data_list[d] = local_data[d]

            if config.verbose:
                print(f"-{task.task_id} Finished: {elapsed} seconds", flush=True)

    except Exception as e:
        print(f"Failed creating Task {task.task_id}: {e}", flush=True)
    finally:
        return


def execute_tasks(taskspaces, tasks: Dict[TaskID, TaskInfo], run_config: RunConfig, data_list=None):

    spawn_start_t = time.perf_counter()

    # Spawn tasks
    for task, details in tasks.items():
        if run_config.movement_type == MovementType.NO_MOVEMENT:
            create_task_no_data(details, taskspaces,
                                config=run_config, data_list=data_list)
        elif run_config.movement_type == MovementType.EAGER_MOVEMENT:
            create_task_eager_data(details, taskspaces,
                                   config=run_config, data_list=data_list)
        elif run_config.movement_type == MovementType.LAZY_MOVEMENT:
            create_task_lazy_data(details, taskspaces,
                                  config=run_config, data_list=data_list)

    spawn_end_t = time.perf_counter()

    return taskspaces


def execute_graph(data_config: Dict[int, DataInfo], tasks: Dict[TaskID, TaskInfo], run_config: RunConfig, timing: List[TimeSample]):

    @spawn(vcus=0, placement=cpu)
    async def main_task():

        graph_times = []

        for i in range(run_config.inner_iterations):
            data_list = generate_data(data_config, run_config.data_scale, run_config.movement_type)

            begin_rl_ts = TaskSpace("begin_rl_task")
            end_rl_ts = TaskSpace("end_rl_task")

            @spawn(begin_rl_ts[0])
            def begin_rl_task():
                pass
                #print("Start RL")
            await begin_rl_ts[0]

            # Initialize task spaces
            taskspaces = {}

            print("Episode ", i, " starts..")
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
            print("Episode :", i, ", elapsed time:", graph_elapsed)
            graph_times.append(graph_elapsed)

            @spawn(end_rl_ts[0])
            def end_rl_task():
                pass
                #print("End RL")
            await end_rl_ts[0]

        graph_times = np.asarray(graph_times)
        graph_t = TimeSample(np.mean(graph_times), np.median(graph_times), np.std(
            graph_times), np.min(graph_times), np.max(graph_times), len(graph_times))

        timing.append(graph_t)


def parse_exec_mode(exec_mode):
    print("Exec mode:", exec_mode)
    if exec_mode == "training":
        return PyMappingPolicyType.RLTraining
    elif exec_mode == "test":
        return PyMappingPolicyType.RLTest
    elif exec_mode == "parla":
        return PyMappingPolicyType.LoadBalancingLocality
    elif exec_mode == "random":
        return PyMappingPolicyType.LoadBalancingLocality
    else:
        print("Unsupported mode:", exec_mode, " so use Parla policy")
        return PyMappingPolicyType.LoadBalancingLocality
 

def run(tasks: Dict[TaskID, TaskInfo], data_config: Dict[int, DataInfo] = None, run_config: RunConfig = None) -> TimeSample:

    if run_config is None:
        run_config = RunConfig(outer_iterations=1, inner_iterations=1,
                               verbose=False, threads=1, data_scale=1)

    timing = []

    exec_mode = parse_exec_mode(run_config.exec_mode)
    for outer in range(run_config.outer_iterations):

        outer_start_t = time.perf_counter()

        with Parla(logfile=run_config.logfile, mapping_policy = exec_mode):
            internal_start_t = time.perf_counter()
            execute_eviction_manager_benchmark(
                data_config, tasks, run_config, timing)
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

    def __init__(self, config: GraphConfig, name: str, graph_path = None, graph_generation = True):
        self.config = config
        self.graph = None
        self.data_config = None

        self.name = name
        self.graph_function = None
        self.graph_generation = graph_generation

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
        if self.graph_generation == True:
            print("Graph write starts")
            with open(self.tmpfilepath, 'w') as tmpfile:
                graph = self.graph_function(self.config)
                #print(graph)
                tmpfile.write(graph)
        else:
            print("Graph read starts")

        self.data_config, self.graph = read_pgraph(self.tmpfilepath)

        return self

    def run(self, run_config: RunConfig, max_time: int = 10000):
        return run(self.graph, self.data_config, run_config)

    def __exit__(self, type, value, traceback):
        self.diro.__exit__(type, value, traceback)


__all__ = [run, verify_order, verify_dependencies,
           verify_complete, verify_time, timeout, GraphContext]
