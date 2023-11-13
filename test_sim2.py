from utility.graphs import *
from utility.load import *

from rich import print

# from utility.execute import run
from utility.visualize import *

from utility.simulator.preprocess import *
from utility.simulator.simulator import *
from utility.simulator.topology import *

from utility.simulator.rl.models.dqn import DQNAgent
from utility.simulator.rl.models.a2c import A2CAgent
from utility.simulator.rl.models.env import *


def run():
    cpu = Device(Architecture.CPU, 0)
    gpu = Device(Architecture.GPU, -1)

    gpu1 = Device(Architecture.GPU, 1)
    gpu2 = Device(Architecture.GPU, 2)

    task_configs = TaskPlacementInfo()

    runtime_info = TaskRuntimeInfo(task_time=100000, device_fraction=1)
    task_configs.add((gpu), runtime_info)

    data_config = DataGraphConfig(pattern=0)

    config = SerialConfig(
        steps=1, chains=100, task_config=task_configs, data_config=data_config
    )
    tasks, data = make_serial_graph(config)
    """

    config = ReductionConfig(task_config=task_configs, data_config=data_config)
    tasks, data = make_reduction_graph(config)
    """

    """

    config = IndependentConfig(task_count=1000, task_config=task_configs, data_config=data_config)
    tasks, data = make_independent_graph(config)
    """

    write_tasks_to_yaml(tasks, "graph")
    write_data_to_yaml(data, "graph")

    topology = TopologyManager.get_generator("frontera")(None)

    rl_environment = ParlaRLNormalizedEnvironment()
    rl_agent = A2CAgent(rl_environment.gcn_indim, rl_environment.fcn_indim,
                        rl_environment.outdim)
    """
    rl_agent = DQNAgent(rl_environment.gcn_indim, rl_environment.fcn_indim,
                        rl_environment.outdim)
    """

    for i in range(20000):
        rl_agent.start_episode()
        tasklist, taskmap, datamap = read_graph("graph")
        sort_tasks_by_order(tasklist, taskmap)
        apply_mapping(taskmap, gpu1)
        print("Epoch:", i)
#print("Epoch:", i, " len:", len(rl_agent.replay_memory))
#print("random ", ExecutionMode.RANDOM)
        scheduler = SimulatedScheduler(topology=topology, rl_mapper=rl_agent, rl_environment=rl_environment,
        execution_mode=ExecutionMode.RANDOM)
        scheduler.register_taskmap(taskmap)
        # scheduler.register_datamap(datamap)
        scheduler.add_initial_tasks(tasklist)
        execution_time = scheduler.run()
        rl_agent.finalize_episode()
        rl_environment.finalize_epoch(execution_time)
run()
