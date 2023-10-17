from utility.graphs import *
from utility.load import *

from rich import print

# from utility.execute import run
from utility.visualize import *

from utility.simulator.preprocess import *
from utility.simulator.simulator import *
from utility.simulator.topology import *


def run():
    cpu = Device(Architecture.CPU, 0)
    gpu = Device(Architecture.GPU, -1)

    gpu1 = Device(Architecture.GPU, 1)
    gpu2 = Device(Architecture.GPU, 2)

    task_configs = TaskPlacementInfo()

    runtime_info = TaskRuntimeInfo(task_time=1000, gil_accesses=1)
    task_configs.add((gpu), runtime_info)

    data_config = DataGraphConfig(pattern=1)

    config = SerialConfig(
        steps=3, chains=3, task_config=task_configs, data_config=data_config
    )

    tasks, data = make_serial_graph(config)

    write_tasks_to_yaml(tasks, "graph")
    write_data_to_yaml(data, "graph")

    tasklist, taskmap, datamap = read_graph("graph")
    apply_mapping(taskmap, gpu1)

    topology = TopologyManager.get_generator("frontera")(None)

    scheduler = SimulatedScheduler(topology=topology)
    scheduler.register_taskmap(taskmap)
    # scheduler.register_datamap(datamap)
    print("Tasklist", tasklist)
    scheduler.add_initial_tasks(tasklist)
    scheduler.run()


run()
