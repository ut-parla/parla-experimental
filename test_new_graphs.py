from utility.graphs import *
from utility.load import *
from rich import print

from utility.execute import run
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
    task_configs.add((cpu, gpu), runtime_info)

    data_config = DataGraphConfig(pattern=1)

    config = SerialConfig(
        steps=3, chains=3, task_config=task_configs, data_config=data_config
    )

    tasks, data = make_serial_graph(config)

    write_tasks_to_yaml(tasks, "graph")
    write_data_to_yaml(data, "graph")

    tasklist, taskmap, datamap = read_graph("graph")

    networkx_graph, networkx_label = build_networkx_graph(taskmap)
    apply_networkx_order(networkx_graph, taskmap)
    print(taskmap)

    # networkx_graph, networkx_label = convert_to_networkx(computetask, datatask)
    # # Convert networkx graph to pydot graph, and export it to png file
    # plot_pydot(networkx_graph)
    # valid_node_order_str = get_valid_order(networkx_graph)
    # print(valid_node_order_str)
    # sorted_task_list = get_sorted_tasks(name_to_task, computetask, datatask, valid_node_order_str)
    # print("task list:", sorted_task_list)

    # hw_topo = create_4gpus_1cpu_hwtopo()
    # scheduler = SimulatedScheduler(topology=hw_topo, task_list=sorted_task_list)
    # scheduler.run()


run()
