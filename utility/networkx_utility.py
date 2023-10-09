import networkx as nx
import pydot
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from itertools import chain

from .types import *
from .simulator.preprocess import *


def convert_to_networkx(computetask: SimulatedTaskMap, datatask: SimulatedTaskMap):
    G = nx.DiGraph()
    labels = {}
    for task in chain(computetask.values(), datatask.values()):
        name = str(task.name)
        if isinstance(task, SimulatedComputeTask):
            labels[name] = name
            G.add_node(name, label=name)
        elif isinstance(task, SimulatedDataTask):
            data_task_label = name + " ["
            data_dependencies = task.info.data_dependencies
            for read_data in data_dependencies.read:
                data_task_label += str(read_data.id) + "(r),"
            for write_data in data_dependencies.write:
                data_task_label += str(write_data.id) + "(w),"
            for rw_data in data_dependencies.write:
                data_task_label += str(rw_data.id) + "(rw),"
            data_task_label = data_task_label[:-1] + "]"
            labels[name] = data_task_label
            G.add_node(name, label=data_task_label)
        for dependency in task.dependencies:
            G.add_edge(str(dependency), name)
    return G, labels


def plot_networkx(G: nx.DiGraph, labels: Dict[TaskID, str]):
    pos = nx.spring_layout(G, seed=5)
    nx.draw_networkx_nodes(G, pos=pos, node_size=700)
    nx.draw_networkx_edges(G, pos=pos)
    nx.draw_networkx_labels(G, pos=pos, labels=labels)
    plt.tight_layout()
    plt.axis("off")


def plot_pydot(G: nx.DiGraph):
    pg = nx.drawing.nx_pydot.to_pydot(G)
    png_str = pg.create_png(prog='dot')
    pg.write_png("pydot_graph.png")
    sio = io.BytesIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    implot = plt.imshow(img, aspect='equal')


def get_valid_order(G: nx.DiGraph):
    """
    Sort a graph by a topology, and return a valid order of the graph.
    """
    nodes = nx.topological_sort(G)
    order = [str(node) for node in nodes]
    return order


def get_sorted_tasks(name_to_task, computetask, datatask, order) -> List[SimulatedTask]:
    """
    Sort SimulatedTask instances by order passed.
    """
    task_list = []
    task_order = 0
    for task_name in order:
        name_to_task[task_name].info.order = task_order
        task_list.append(name_to_task[task_name])
        task_order += 1
    return task_list
