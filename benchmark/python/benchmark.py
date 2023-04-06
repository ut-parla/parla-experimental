import time
import sys

from parla.utility.graphs import DeviceType, TaskConfig, TaskConfigs
from parla.utility.graphs import IndependentConfig, SerialConfig, ReductionConfig, RunConfig
from parla.utility.graphs import DataInitType
from parla.utility.graphs import read_pgraph, parse_blog
from parla.utility.graphs import shuffle_tasks
from parla.utility.graphs import MovementType

from parla.utility.execute import verify_order, verify_dependencies, verify_complete, verify_time, verify_states
from parla.utility.execute import GraphContext

import argparse

parser = argparse.ArgumentParser(description='Launch graph file in Parla')
parser.add_argument('-d', metavar='N', type=int, help='The dimension of data segments >=2 (Increase to make movement more expensive)', default=2)
parser.add_argument('-width_bytes', metavar='N', type=int, help='Bytes of fD data array', default=2)
parser.add_argument('-data_move', metavar='data_move', type=int, help='type of data movement. options=(None=0, Lazy=1, Eager=2)', default=0)
parser.add_argument('-graph', metavar='graph', type=str, help='the input graph file to run', required=True, default='graph/independent.gph')
parser.add_argument('-user', metavar='user', type=int, help='type of placement. options=(None=0, User=1)', default=0)
parser.add_argument('-computation_weight', metavar='computation_weight', type=int, help='length of task compute time', default=None)
parser.add_argument('-n', metavar='n', type=int, help='number of tasks', default=None)
parser.add_argument('-gil_count', metavar='gil_count', type=int, help='number of gil count', default=1)
parser.add_argument('-gil_time', metavar='gil_time', type=int, help='gil time', default=100)
parser.add_argument('-overlap', metavar='overlap', type=int, help='1 if data is overlapped otherwise 0', default=1)
parser.add_argument('-workloads', metavar='workloads', type=str, help='workload types (reduction, independent, serial)', default=None)
parser.add_argument('-level', metavar='level', type=int, help='for reduction, level of a reduction tree', default=8)
parser.add_argument('-branch', metavar='branch', type=int, help='for reduction, branch', default=2)
parser.add_argument('-num_gpus', metavar='num_gpus', type=int, help='the number of gpus', default=4)

args = parser.parse_args()

#movement_type = MovementType.NO_MOVEMENT
#movement_type = MovementType.EAGER_MOVEMENT
data_scale = 1

fixedp = True

def parse_user_chosen_placement_flag(flag, num_gpus):
    """
    Return a pair of a boolean value and the target device type.
    The first boolean is corresponding to the second element of
    the pair and true if user-chosen device mode is enabled.
    Otherwise return false.
    """
    if flag == 1:
        return (True, DeviceType.USER_CHOSEN_DEVICE)
    else:
        if num_gpus == 0:
            return (False, DeviceType.CPU_DEVICE)
        else:
            return (False, DeviceType.ANY_GPU_DEVICE)


def parse_data_pattern_flag(flag):
    if flag == 0:
        return DataInitType.INDEPENDENT_DATA
    elif flag == 1:
        return DataInitType.OVERLAPPED_DATA
    else:
        raise ValueError(f"Incorrect data overlap flag: {flag}")


def parse_data_move_flag(flag):
    if flag == 0:
        return MovementType.NO_MOVEMENT
    elif flag == 1:
        return MovementType.LAZY_MOVEMENT
    elif flag == 2:
        return MovementType.EAGER_MOVEMENT
    else:
        raise ValueError(f"Incorrect data move flag: {flag}")


def reduction_scalinum_gpus(fD_array_bytes, sD_array_bytes, \
        num_gpus, fixed_place, user_chosen_device, computation_weight, \
        gil_count, gil_time, data_pattern, level, branch, data_move_type, \
        graph_path):
    print(f"[Reduction] fD array bytes: {fD_array_bytes}, " +
          f"sD array bytes: {sD_array_bytes} " +
          f"Num GPUs: {num_gpus}, Fixed Placement: {fixed_place}, " +
          f"User flag: {user_chosen_device}, " +
          f"Computation Weight: {computation_weight} " +
          f"Level: {level}, Branch: {branch}, GIL count: {gil_count}, " +
          f"GIL time: {gil_time}, Data overlap: " +
          f"{data_pattern} Data move type: {data_move_type}" +
          f" Graph path: {graph_path}")
    device_fraction = 1.0
    if num_gpus == 0: 
        concurrent_tasks = num_gpus
        device_fraction = 1.0 / concurrent_tasks
        user_chosen_device = DeviceType.CPU_DEVICE

    # Configuration for task graph generation
    task_configs = TaskConfigs()
    task_configs.add(user_chosen_device, TaskConfig(
        task_time=computation_weight, gil_accesses=gil_count,
        gil_fraction=gil_time, device_fraction=device_fraction))
    config = ReductionConfig(data_pattern=data_pattern,
        fixed_placement=fixed_place, num_gpus=num_gpus,
        total_data_width=fD_array_bytes, levels=level, branch_factor=branch,
        task_config=task_configs)

    with GraphContext(config, name="reduction", graph_path=graph_path) as g:

        logpath = g.tmplogpath

        run_config = RunConfig(
            outer_iterations=1,
            inner_iterations=1,
            verbose=False,
            logfile=logpath,
            num_gpus=num_gpus,
            movement_type=data_move_type,
            data_scale=sD_array_bytes)

        times = g.run(run_config)
        log_times, log_graph, log_states = parse_blog(logpath)
        assert (verify_complete(log_graph, g.graph))
        assert (verify_dependencies(log_graph, g.graph))
        #assert (verify_order(log_times,g.graph))
        assert (verify_states(log_states))

    print(f"new_parla,reduction,{branch}-{reduction},{num_gpus},{fixed_place},{fD_array_bytes},{data_move_type},"+
          f"{times.mean}", flush=True)


def independent_scalinum_gpus(fD_array_bytes, sD_array_bytes, num_gpus,  \
        fixed_place, user_chosen_device, computation_weight, num_tasks, gil_count,  \
        gil_time, data_pattern, data_move_type, graph_path):
    print(f"[Indp] fD array bytes: {fD_array_bytes}, " +
          f"sD array bytes: {sD_array_bytes} Num GPUs: {num_gpus}, " +
          f"Fixed place: {fixed_place}, User flag: {user_chosen_device}, " +
          f"Computation Weight: {computation_weight} Num tasks: {num_tasks}, " +
          f"GIL count: {gil_count}, GIL time: {gil_time}, Data overlap: " +
          f"{data_pattern} Data move type: {data_move_type}" +
          f" Graph path: {graph_path}")
    device_fraction = 1.0
    if num_gpus == 0:
        concurrent_tasks = num_gpus
        device_fraction = 1.0 / concurrent_tasks
        user_chosen_device = DeviceType.CPU_DEVICE

    # Configuration for task graph generation
    task_configs = TaskConfigs()
    task_configs.add(user_chosen_device, TaskConfig(
        task_time=computation_weight, gil_accesses=gil_count,
        gil_fraction=gil_time, device_fraction=device_fraction))
    config = IndependentConfig(data_pattern=data_pattern,
        fixed_placement=fixed_place, num_gpus=num_gpus,
        total_data_width=fD_array_bytes, task_count=num_tasks,
        task_config=task_configs)

    with GraphContext(config, name="independent", graph_path=graph_path) as g:

        logpath = g.tmplogpath

        run_config = RunConfig(
            outer_iterations=1,
            inner_iterations=1,
            verbose=False,
            logfile=logpath,
            num_gpus=num_gpus,
            movement_type=data_move_type,
            data_scale=sD_array_bytes)

        times = g.run(run_config)
        log_times, log_graph, log_states = parse_blog(logpath)
        assert (verify_complete(log_graph, g.graph))
        assert (verify_dependencies(log_graph, g.graph))
        #assert (verify_order(log_times,g.graph))
        assert (verify_states(log_states))

    print(f"new_parla,independent,{num_tasks},{num_gpus},{fixed_place},{fD_array_bytes},{data_move_type},"+
          f"{times.mean}", flush=True)

def serial_scalinum_gpus(fD_array_bytes, sD_array_bytes, num_gpus,
        fixed_place, user_chosen_device, computation_weight, num_tasks, gil_count,  \
        gil_time, data_pattern, data_move_type, graph_path):
    print(f"[Serial] fD array bytes: {fD_array_bytes}, " +
          f"sD array bytes: {sD_array_bytes} Num GPUs: {num_gpus}, " +
          f"Fixed place: {fixed_place}, User flag: {user_chosen_device}, " +
          f"Computation Weight: {computation_weight} " +
          f"Num tasks: {num_tasks}, GIL count: {gil_count}, " +
          f"GIL time: {gil_time}, Data overlap: " +
          f"{data_pattern} Data move type: {data_move_type}" +
          f" Graph path: {graph_path}")
    device_fraction = 1.0
    concurrent_tasks = num_gpus
    if num_gpus == 0:
        device_fraction = 1.0 / concurrent_tasks
        user_chosen_device = DeviceType.CPU_DEVICE

    task_configs = TaskConfigs()
    task_configs.add(user_chosen_device, TaskConfig(
        task_time=computation_weight, gil_accesses=gil_count,
        gil_fraction=gil_time, device_fraction=device_fraction))
    config = SerialConfig(data_pattern=data_pattern,
        total_data_width=fD_array_bytes, steps=num_tasks, chains=num_gpus,
        task_config=task_configs,
        num_gpus=num_gpus, fixed_placement=fixed_place)

    with GraphContext(config, name="serial", graph_path=graph_path) as g:

        logpath = g.tmplogpath

        run_config = RunConfig(
            outer_iterations=1,
            inner_iterations=1,
            verbose=False,
            num_gpus=num_gpus,
            logfile=logpath,
            movement_type=data_move_type,
            data_scale=sD_array_bytes)

        times = g.run(run_config)
        log_times, log_graph, log_states = parse_blog(logpath)
        assert (verify_complete(log_graph, g.graph))
        assert (verify_dependencies(log_graph, g.graph))
        #assert (verify_order(log_times,g.graph))
        assert (verify_states(log_states))

    print(f"new_parla,serial,{num_tasks},{num_gpus},{fixed_place},{fD_array_bytes},{data_move_type},"+
          f"{times.mean}", flush=True)


if __name__ == "__main__":
    options = f"fD dimension: {args.width_bytes} " + \
          f"sD Dimension: {args.d} Data move: {args.data_move} Graph: {args.graph}" +  \
          f" User: {args.user} Computation weight: {args.computation_weight} "+  \
          f" # of tasks: {args.n} Gil count: {args.gil_count} Gil time: {args.gil_time} "+  \
          f" Data overlapped?: {args.overlap} Workload type: {args.workloads} "+  \
          f" Level for reduction: {args.level} Branch for reduction: {args.branch}" + \
          f" Number of GPUs: {args.num_gpus}"
    fD_array_bytes = args.width_bytes
    sD_array_bytes = args.d
    num_gpus = args.num_gpus
    fixed_place, user_chosen_device = parse_user_chosen_placement_flag(args.user, num_gpus)
    computation_weight = args.computation_weight
    num_tasks = args.n
    gil_count = args.gil_count
    gil_time = args.gil_time
    data_pattern = parse_data_pattern_flag(args.overlap)
    reduce_level = args.level
    reduce_branch = args.branch
    graph_type = args.workloads
    data_move_type = parse_data_move_flag(args.data_move)
    graph_path = args.graph

    if fD_array_bytes == 0:
        data_pattern = DataInitType.NO_DATA

    if graph_type == "reduction":
        reduction_scalinum_gpus(fD_array_bytes, sD_array_bytes,
            num_gpus, fixed_place, user_chosen_device, computation_weight, gil_count,
            gil_time, data_pattern, reduce_level, reduce_branch,
            data_move_type, graph_path)
    elif graph_type == "independent":
        independent_scalinum_gpus(fD_array_bytes, sD_array_bytes,
            num_gpus, fixed_place, user_chosen_device, computation_weight, num_tasks,
            gil_count, gil_time, data_pattern, data_move_type, graph_path)
    elif graph_type == "serial":
        serial_scalinum_gpus(fD_array_bytes, sD_array_bytes,
            num_gpus, fixed_place, user_chosen_device, computation_weight, num_tasks,
            gil_count, gil_time, data_pattern, data_move_type, graph_path)
    else:
        raise ValueError(f"Does not support this workload type: {graph_type} (Supporting reduction, independent, serial)")
