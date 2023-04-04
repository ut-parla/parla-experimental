import time

from parla.utility.graphs import DeviceType, TaskConfig, TaskConfigs
from parla.utility.graphs import IndependentConfig, SerialConfig, ReductionConfig, RunConfig
from parla.utility.graphs import DataInitType
from parla.utility.graphs import read_pgraph, parse_blog
from parla.utility.graphs import shuffle_tasks
from parla.utility.graphs import MovementType

from parla.utility.execute import verify_order, verify_dependencies, verify_complete, verify_time, verify_states
from parla.utility.execute import GraphContext

movement_type = MovementType.NO_MOVEMENT
#movement_type = MovementType.LAZY_MOVEMENT
#movement_type = MovementType.EAGER_MOVEMENT
data_scale = 1

num_gpus = 4

def serial_scaling():
    for ng in range(1, num_gpus + 1):
        for fixedp in (True, False):
            max_time = 10
            task_time = 16000
            n = 300
            device_type = DeviceType.USER_CHOSEN_DEVICE
            cost = 1.0
            concurrent_tasks = ng
            
            if ng == 0:
                cost = 1.0 / concurrent_tasks
                device_type = DeviceType.CPU_DEVICE

            task_configs = TaskConfigs()
            task_configs.add(device_type, TaskConfig(
                task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))

            config = SerialConfig(data_pattern=DataInitType.OVERLAPPED_DATA,
                total_data_width=6250, steps=n, chains=1, task_config=task_configs,
                num_gpus=ng, fixed_placement=fixedp)

            with GraphContext(config, name="serial") as g:

                logpath = g.tmplogpath

                run_config = RunConfig(
                    outer_iterations=1,
                    inner_iterations=1,
                    verbose=False,
                    num_gpus=ng,
                    logfile=logpath,
                    movement_type=movement_type,
                    data_scale=data_scale)

                timing = g.run(run_config, max_time=max_time)
                log_times, log_graph, log_states = parse_blog(logpath)
                assert (verify_complete(log_graph, g.graph))
                assert (verify_dependencies(log_graph, g.graph))
                assert (verify_order(log_times,g.graph))
                assert (verify_states(log_states))

            print("[serial] # gpus:", ng, " fixed:", fixedp, " mean time:", timing.mean, flush=True)


def independent_scaling():
    for ng in range(1, num_gpus + 1):
        for fixedp in (True, False):

            max_time = 50
            task_time = 16000
            n = 3000
            device_type = DeviceType.ANY_GPU_DEVICE
            cost = 1.0
            if ng == 0:
                concurrent_tasks = ng
                cost = 1.0 / concurrent_tasks
                device_type = DeviceType.CPU_DEVICE

            # Configuration for task graph generation
            task_configs = TaskConfigs()
            task_configs.add(device_type, TaskConfig(
                task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))
            config = IndependentConfig(data_pattern=DataInitType.NO_DATA,
                total_data_width=6250, task_count=n, task_config=task_configs, num_gpus=ng,
                fixed_placement=fixedp)

            with GraphContext(config, name="independent") as g:

                logpath = g.tmplogpath

                run_config = RunConfig(
                    outer_iterations=1,
                    inner_iterations=1,
                    verbose=False,
                    logfile=logpath,
                    num_gpus=ng,
                    movement_type=movement_type,
                    data_scale=data_scale)

                timing = g.run(run_config, max_time=max_time)
                log_times, log_graph, log_states = parse_blog(logpath)
                assert (verify_complete(log_graph, g.graph))
                assert (verify_dependencies(log_graph, g.graph))
                assert (verify_order(log_times,g.graph))
                assert (verify_states(log_states))

            print("[indp] # gpus:", ng, " fixed:", fixedp, " mean time:", timing.mean, flush=True)


def reduction_scaling():
    for ng in range(1, num_gpus + 1):
        for fixedp in [True]:
            max_time = 100
            task_time = 16000
            device_type = DeviceType.USER_CHOSEN_DEVICE
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
                fixed_placement=fixedp,
                total_data_width=6250, levels=8, branch_factor=2,
                task_config=task_configs, num_gpus=num_gpus)

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

                timing = g.run(run_config, max_time=max_time)
                log_times, log_graph, log_states = parse_blog(logpath)
                assert (verify_complete(log_graph, g.graph))
                assert (verify_dependencies(log_graph, g.graph))
                assert (verify_order(log_times,g.graph))
                assert (verify_states(log_states))

            print("[reduction] # gpus:", ng, " fixed:", fixedp, " mean time:", timing.mean, flush=True)


if __name__ == "__main__":
    independent_scaling()
    serial_scaling()
    reduction_scaling()
