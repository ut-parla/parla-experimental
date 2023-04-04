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

# movement_type = MovementType.NO_MOVEMENT
# movement_type = MovementType.EAGER_MOVEMENT
data_scale = 1

num_gpus = int(sys.argv[1])

fixedp = True
if sys.argv[2] == "fixed-placement":
    fixedp = True
elif sys.argv[2] == "policy":
    fixedp = False

movement_type = MovementType.LAZY_MOVEMENT
if sys.argv[3] == "no":
    movement_type = MovementType.NO_MOVEMENT
elif sys.argv[3] == "lazy":
    movement_type = movement_type.LAZY_MOVEMENT
elif sys.argv[3] == "eager":
    movement_type = movement_type.EAGER_MOVEMENT

total_data_width = 62500
n = 1000
task_time = 5000
max_time = 100


def serial_scalinum_gpus():
    device_type = DeviceType.USER_CHOSEN_DEVICE
    cost = 1.0
    concurrent_tasks = num_gpus
    if num_gpus == 0:
        cost = 1.0 / concurrent_tasks
        device_type = DeviceType.CPU_DEVICE

    task_configs = TaskConfigs()
    task_configs.add(device_type, TaskConfig(
        task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))

    config = SerialConfig(data_pattern=DataInitType.OVERLAPPED_DATA,
                          total_data_width=total_data_width, steps=n, chains=1, task_config=task_configs,
                          num_gpus=num_gpus, fixed_placement=fixedp)

    with GraphContext(config, name="serial") as g:

        logpath = g.tmplogpath

        run_config = RunConfig(
            outer_iterations=1,
            inner_iterations=1,
            verbose=False,
            num_gpus=num_gpus,
            logfile=logpath,
            movement_type=movement_type,
            data_scale=data_scale)

        timinum_gpus = g.run(run_config, max_time=max_time)
        log_times, log_graph, log_states = parse_blog(logpath)
        assert (verify_complete(log_graph, g.graph))
        assert (verify_dependencies(log_graph, g.graph))
        assert (verify_order(log_times, g.graph))
        assert (verify_states(log_states))

    print("serial, # gpus,", num_gpus, ", fixed,",
          sys.argv[2], ", data,", sys.argv[3], " mean time:", timinum_gpus.mean, flush=True)


def independent_scalinum_gpus():
    device_type = DeviceType.ANY_GPU_DEVICE
    cost = 1.0
    if num_gpus == 0:
        concurrent_tasks = num_gpus
        cost = 1.0 / concurrent_tasks
        device_type = DeviceType.CPU_DEVICE

    # Configuration for task graph generation
    task_configs = TaskConfigs()
    task_configs.add(device_type, TaskConfig(
        task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))
    config = IndependentConfig(data_pattern=DataInitType.NO_DATA,
                               total_data_width=total_data_width, task_count=n, task_config=task_configs, num_gpus=num_gpus,
                               fixed_placement=fixedp)

    with GraphContext(config, name="independent") as g:

        logpath = g.tmplogpath

        run_config = RunConfig(
            outer_iterations=1,
            inner_iterations=1,
            verbose=False,
            logfile=logpath,
            num_gpus=num_gpus,
            movement_type=movement_type,
            data_scale=data_scale)

        timinum_gpus = g.run(run_config, max_time=max_time)
        log_times, log_graph, log_states = parse_blog(logpath)
        assert (verify_complete(log_graph, g.graph))
        assert (verify_dependencies(log_graph, g.graph))
        assert (verify_order(log_times, g.graph))
        assert (verify_states(log_states))

    print("independent, # gpus,", num_gpus, ", fixed,",
          sys.argv[2], ", data,", sys.argv[3], " mean time:", timinum_gpus.mean, flush=True)


def reduction_scalinum_gpus():
    device_type = DeviceType.ANY_GPU_DEVICE
    cost = 1.0
    if num_gpus == 0:
        concurrent_tasks = num_gpus
        cost = 1.0 / concurrent_tasks
        device_type = DeviceType.CPU_DEVICE

    # Configuration for task graph generation
    task_configs = TaskConfigs()
    task_configs.add(device_type, TaskConfig(
        task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))
    config = ReductionConfig(data_pattern=DataInitType.OVERLAPPED_DATA,
                             fixed_placement=fixedp, num_gpus=num_gpus,
                             total_data_width=total_data_width, levels=9, branch_factor=2,
                             task_config=task_configs)

    with GraphContext(config, name="reduction") as g:

        logpath = g.tmplogpath

        run_config = RunConfig(
            outer_iterations=1,
            inner_iterations=1,
            verbose=False,
            logfile=logpath,
            num_gpus=num_gpus,
            movement_type=movement_type,
            data_scale=data_scale)

        timinum_gpus = g.run(run_config, max_time=max_time)
        log_times, log_graph, log_states = parse_blog(logpath)
        assert (verify_complete(log_graph, g.graph))
        assert (verify_dependencies(log_graph, g.graph))
        assert (verify_order(log_times, g.graph))
        assert (verify_states(log_states))

    print("reduction, # gpus,", num_gpus, ", fixed,",
          sys.argv[2], ", data,", sys.argv[3], " mean time:", timinum_gpus.mean, flush=True)


if __name__ == "__main__":
    independent_scalinum_gpus()
    serial_scalinum_gpus()
    reduction_scalinum_gpus()
