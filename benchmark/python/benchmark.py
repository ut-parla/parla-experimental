import time
import google_benchmark as benchmark

from parla.utility.graphs import DeviceType, TaskConfig, TaskConfigs
from parla.utility.graphs import IndependentConfig, SerialConfig, ReductionConfig, RunConfig
from parla.utility.graphs import DataInitType
from parla.utility.graphs import read_pgraph, parse_blog
from parla.utility.graphs import shuffle_tasks
from parla.utility.graphs import MovementType

from parla.utility.execute import verify_order, verify_dependencies, verify_complete, verify_time
from parla.utility.execute import GraphContext

movement_type = MovementType.LAZY_MOVEMENT
#movement_type = MovementType.EAGER_MOVEMENT
data_scale = 1

@benchmark.register(name="SerialScaling")
@benchmark.option.args_product([(1, 2, 4, 8), (1, 2)]) # TODO(hc): will add data pattern too
def serial_scaling(state):
    while state:
        max_time = 10
        task_time = 1000
        n = 1000
        use_gpus = True if state.range(1) == 1 else False
        print("GPU mode:", use_gpus)
        device_type = DeviceType.USER_CHOSEN_DEVICE
        cost = 1.0
        concurrent_tasks = state.range(0)
        if not use_gpus:
            cost = 1.0 / concurrent_tasks
            device_type = DeviceType.CPU_DEVICE

        task_configs = TaskConfigs()
        task_configs.add(device_type, TaskConfig(
            task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))

        config = SerialConfig(data_pattern=DataInitType.OVERLAPPED_DATA,
            total_data_width=6250, steps=n, chains=concurrent_tasks, task_config=task_configs)

        with GraphContext(config, name="serial") as g:

            logpath = g.tmplogpath

            run_config = RunConfig(
                outer_iterations=1,
                inner_iterations=1,
                verbose=False,
                logfile=logpath,
                movement_type=movement_type,
                data_scale=data_scale)

            timing = g.run(run_config, max_time=max_time)

        state.set_iteration_time(timing.mean)


#@benchmark.register(name="IndependentScaling")
@benchmark.option.args_product([(1, 2, 4, 8), (1, 2)]) # TODO(hc): will add data pattern too
def independent_scaling(state):
    while state:
        max_time = 10
        task_time = 1000
        n = 1000
        use_gpus = True if state.range(1) else False
        print("GPU mode:", use_gpus)
        device_type = DeviceType.ANY_GPU_DEVICE
        cost = 1.0
        if not use_gpus:
            concurrent_tasks = state.range(0)
            cost = 1.0 / concurrent_tasks
            device_type = DeviceType.CPU_DEVICE

        # Configuration for task graph generation
        task_configs = TaskConfigs()
        task_configs.add(device_type, TaskConfig(
            task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))
        config = IndependentConfig(data_pattern=DataInitType.INDEPENDENT_DATA,
            total_data_width=6250, task_count=n, task_config=task_configs, use_gpus=use_gpus)
        with GraphContext(config, name="independent") as g:

            logpath = g.tmplogpath

            run_config = RunConfig(
                outer_iterations=1,
                inner_iterations=1,
                verbose=False,
                logfile=logpath,
                movement_type=movement_type,
                data_scale=data_scale)

            timing = g.run(run_config, max_time=max_time)

        state.set_iteration_time(timing.mean)


#@benchmark.register(name="ReductionScaling")
@benchmark.option.args_product([(1, 2, 4, 8), (1, 2)]) # TODO(hc): will add data pattern too
def reduction_scaling(state):
    while state:
        max_time = 100
        task_time = 1000
        n = 1000
        use_gpus = True if state.range(1) == 1 else False
        print("GPU mode:", use_gpus)
        device_type = DeviceType.USER_CHOSEN_DEVICE
        cost = 1.0
        if not use_gpus:
            concurrent_tasks = state.range(0)
            cost = 1.0 / concurrent_tasks
            device_type = DeviceType.CPU_DEVICE

        # Configuration for task graph generation
        task_configs = TaskConfigs()
        task_configs.add(device_type, TaskConfig(
            task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))
        config = ReductionConfig(data_pattern=DataInitType.OVERLAPPED_DATA,
            fixed_placement=True,
            total_data_width=6250, levels=8, branch_factor=2, task_config=task_configs, use_gpus=use_gpus)

        with GraphContext(config, name="reduction") as g:

            logpath = g.tmplogpath

            run_config = RunConfig(
                outer_iterations=1,
                inner_iterations=1,
                verbose=False,
                logfile=logpath,
                movement_type=movement_type,
                data_scale=data_scale)

            timing = g.run(run_config, max_time=max_time)

        state.set_iteration_time(timing.mean)


if __name__ == "__main__":
    benchmark.main()
