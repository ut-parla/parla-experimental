import time
import google_benchmark as benchmark

from parla.utility.graphs import DeviceType, TaskConfig, TaskConfigs
from parla.utility.graphs import IndependentConfig, SerialConfig, RunConfig
from parla.utility.graphs import read_pgraph, parse_blog
from parla.utility.graphs import shuffle_tasks

from parla.utility.execute import verify_order, verify_dependencies, verify_complete, verify_time
from parla.utility.execute import GraphContext

#@benchmark.register(name="SerialScaling")
@benchmark.option.range_multiplier(2)
@benchmark.option.range(1, 8)
def serial_scaling(state):
    while state:
        max_time = 10
        task_time = 1000
        n = 1000
        concurrent_tasks = state.range(0)
        cost = 1.0 / concurrent_tasks

        task_configs = TaskConfigs()
        task_configs.add(DeviceType.CPU_DEVICE, TaskConfig(
            task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))

        config = SerialConfig(
            steps=n, chains=concurrent_tasks, task_config=task_configs)

        with GraphContext(config, name="serial") as g:

            logpath = g.tmplogpath

            run_config = RunConfig(
                outer_iterations=1,
                inner_iterations=1,
                verbose=False,
                logfile=logpath)

            timing = g.run(run_config, max_time=max_time)

        state.set_iteration_time(timing.mean)


@benchmark.register(name="IndependentScaling")
@benchmark.option.args_product([(1, 2, 4, 8), (1, 2)]) # TODO(hc): will add data pattern too
def independent_scaling(state):
    while state:
        max_time = 10
        task_time = 1000
        n = 1000
        use_gpus = True if state.range(1) else False
        device_type = DeviceType.ANY_GPU_DEVICE
#device_type = DeviceType.CPU_DEVICE
        cost = 1.0
        if not use_gpus:
            concurrent_tasks = state.range(0)
            cost = 1.0 / concurrent_tasks
            device_type = DeviceType.CPU_DEVICE

        # Configuration for task graph generation
        task_configs = TaskConfigs()
        task_configs.add(device_type, TaskConfig(
            task_time=task_time, gil_accesses=1, gil_fraction=0, device_fraction=cost))
        config = IndependentConfig(task_count=n, task_config=task_configs, use_gpus=use_gpus)
        with GraphContext(config, name="independent") as g:

            logpath = g.tmplogpath

            run_config = RunConfig(
                outer_iterations=1,
                inner_iterations=1,
                verbose=False,
                logfile=logpath)

            timing = g.run(run_config, max_time=max_time)

        state.set_iteration_time(timing.mean)

if __name__ == "__main__":
    benchmark.main()
