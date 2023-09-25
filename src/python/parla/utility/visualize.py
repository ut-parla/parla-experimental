import networkx as nx
import pydot
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from .types import *


def convert_to_networkx(tasks: TaskMap):
    G = nx.DiGraph()
    for task in tasks.values():
        name = task_id_to_str(task.id)
        G.add_node(task.id, label=name)
        for dependency in task.dependencies:
            G.add_edge(dependency, task.id)

    labels = {}
    for task in tasks.values():
        name = task_id_to_str(task.id)
        labels[task.id] = name

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
    sio = io.BytesIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    implot = plt.imshow(img, aspect='equal')
