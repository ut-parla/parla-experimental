import PIL.Image
import PIL.ImageTk
from tkinter import *
from re import S
import matplotlib.pyplot as plt
from fractions import Fraction
import matplotlib.pyplot as plt

def str_to_tuple(string):
    if "M" in string:
        return string
    return eval(string)


def extract(string):
    """
    Extracts string as decimal or int
    """
    if "." in string:
        return Fraction(string)
    else:
        return int(string)


def read_graphx(filename):
    """
    Reads a graphx file and returns:
    1. A list of the nodes in the graph
    2. The initial data configuration
    """

    task_list = []
    data_config = dict()

    with open(filename, 'r') as graph_file:

        lines = graph_file.readlines()

        # Read the initial data configuration
        data_info = lines.pop(0)
        data_info = data_info.split(',')
        # print(data_info)
        idx = 0
        for data in data_info:
            info = data.strip().strip("{}").strip().split(":")
            size = int(info[0].strip())
            location = int(info[1].strip())
            data_config[idx] = (size, location)
            idx += 1

        # print("Data Config", data_config)
        # Read the task graph
        for line in lines:

            task = line.split("|")
            # Breaks into [task_id, task_runtime, task_dependencies, data_dependencies]

            # Process task id (can't be empty)
            task_ids = task[0].strip().split(",")
            task_ids = [int(task_id.strip()) for task_id in task_ids]
            # print("Working on task: ", task_ids)

            # Process task runtime (can't be empty)
            configurations = task[1].strip().split("},")
            task_runtime = dict()
            for config in configurations:
                config = config.strip().strip("{}").strip()
                config = config.split(":")
                target = int(config[0].strip())
                details = config[1].strip().split(",")
                details = [extract(detail.strip()) for detail in details]
                # print(target, details)

                task_runtime[target] = details

            # Process task dependencies (can be empty)
            if len(task) > 2:
                dependencies = task[2].split(":")
                if (len(dependencies) > 0) and (not dependencies[0].isspace()):
                    task_dependencies = []

                    for i in range(len(dependencies)):
                        if not dependencies[i].isspace():
                            ids = dependencies[i].strip().split(",")
                            task_dependencies.append(
                                tuple([int(idx) for idx in ids]))
                else:
                    task_dependencies = []

            else:
                task_dependencies = []

            # Process data dependencies (can be empty)
            if len(task) > 3:
                # print("All Data: ", task[3])
                types = task[3].split(":")
                # Split into [read, write, read/write]

                check = [not t.isspace() for t in types]
                # print("Check: ", check)
                if any(check):
                    task_data = [None, None, None]

                    for i in range(len(types)):
                        if check[i]:
                            data = types[i].strip().split(",")
                            if not data[0].isspace():
                                task_data[i] = [0 for _ in range(len(data))]

                                for j in range(len(data)):
                                    if not data[j].isspace():
                                        task_data[i][j] = int(data[j])
                else:
                    task_data = [None, None, None]
            else:
                task_data = [None, None, None]

            # print("Task Data: ", task_data)
            task_tuple = (task_ids, task_runtime, task_dependencies, task_data)
            task_list.append(task_tuple)

    return data_config, task_list


def convert_to_dictionary(task_list):

    runtime_dict = dict()
    dependency_dict = dict()
    write_dict = dict()
    read_dict = dict()
    count_dict = dict()

    count = 0
    for task in task_list:
        ids, runtime, dependencies, data = task

        # tuple_dep = [] if dependencies[0] is None else [
        #    tuple(idx) for idx in dependencies]
        runtime_dict[tuple(ids)] = runtime
        dependency_dict[tuple(ids)] = dependencies

        list_in = [] if data[0] is None else [k for k in data[0]]
        list_out = [] if data[1] is None else [k for k in data[1]]
        list_rw = [] if data[2] is None else [k for k in data[2]]

        write_dict[tuple(ids)] = list_out+list_rw
        read_dict[tuple(ids)] = list_in+list_rw

        count += 1
        count_dict[tuple(ids)] = count

    return runtime_dict, dependency_dict, write_dict, read_dict, count_dict


def bfs(graph, node, target, writes):
    """
    Return last task to touch the data
    """
    queue = []
    visited = []
    visited.append(node)
    queue.append(node)

    while queue:
        s = queue.pop(0)
        # print("Visiting: ", graph[s])
        for neighbor in graph[s]:

            writes_to = writes[neighbor]

            if target in writes_to:
                return neighbor if neighbor != node else None

            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)

    return None


def compute_data_edges(graph_dictionaries):
    """
    For each task, compute the dependencies for each of its data terms.
    E.g. find the most recent write to each of its inputs
    """
    task_data_dependencies = dict()

    runtime_dict, dependency_dict, write_dict, read_dict, count_dict = graph_dictionaries

    for current_task, dependencies in dependency_dict.items():
        reads = read_dict[current_task]
        writes = write_dict[current_task]

        touches = set(reads)

        data_dependencies = dict()

        for target in touches:
            # print("Target: ", target, "Current: ",
            #      current_task, "Dependencies: ", dependencies)
            last_write = bfs(dependency_dict, current_task, target, write_dict)
            if last_write is None:
                # There is no dependency to this read. It is possibly initial.
                # Label it as an endpoint node.
                last = f"D{target}"
                data_dependencies[target] = last
            else:
                # This data has been written to before.
                # Label it as a data dependency.
                data_dependencies[target] = last_write

        task_data_dependencies[current_task] = data_dependencies

    return task_data_dependencies


def add_data_tasks(task_list, task_dictionaries, task_data_dependencies):
    """
    Create a list of data tasks.
    Create a dictionary of data tasks (keyed on the tasks that spawned them).
    """
    runtime_dict, dependency_dict, write_dict, read_dict, count_dict = task_dictionaries

    n_tasks = len(task_list)
    data_tasks = []
    datamove_task_meta_info = dict()
    compute_tid_to_datamove_tid = dict()

    count = n_tasks

    for task in task_list:
        # For each task create all data movement tasks
        ids, runtime, dependencies, data = task
        data_dependencies = task_data_dependencies[tuple(ids)]
        compute_tid_to_datamove_tid[tuple(ids)] = []

        # Make one data movement task for each piece of data that needs to be read
        for data in read_dict[tuple(ids)]:
            movement_id = f"M({str(data)}, {tuple(ids)})"
            movement_depenendices = []

            # Create list of task dependencies for data movement task
            if data in data_dependencies:
                # print("Data: ", data, "Dependencies: ",
                #      data_dependencies[data])
                all_data_dependencies = data_dependencies[data]
                all_data_dependencies = [all_data_dependencies] if type(
                    all_data_dependencies) is not list else all_data_dependencies
                for dependency in all_data_dependencies:
                    if "D" not in dependency:
                        movement_depenendices.append(dependency)
            else:
                raise Exception(
                    "Preprocessing Failed. Data not found in data dependencies")

            # Add data movement task to parent dependencies
            dependencies.append(movement_id)

            # Create data movement task tuple
            data_task_tuple = (movement_id, movement_depenendices, [data])
            data_tasks.append(data_task_tuple)
            datamove_task_meta_info[movement_id] = data_task_tuple
            compute_tid_to_datamove_tid[tuple(ids)].append(movement_id)
            count += 1
            count_dict[movement_id] = count

    return data_tasks, datamove_task_meta_info, compute_tid_to_datamove_tid


def make_networkx_graph(task_list, task_dictionaries, data_config, movement_dictionaries=None, plot_isolated=False, plot_weights=False, check_redundant=True):
    """
    Create a networkx graph from the task list.
    """
    import networkx as nx
    graph = nx.DiGraph()

    # movement_dictionaries = None
    runtime_dict, dependency_dict, write_dict, read_dict, count_dict = task_dictionaries

    if movement_dictionaries is not None:
        data_tasks, datamove_task_meta_info, compute_tid_to_datamove_tid = movement_dictionaries

    for task in task_list:
        ids, runtime, dependencies, data = task
        # print("Runtime: ", (runtime))
        # print("Data: ", data)
        graph.add_node(tuple(ids), data=data)

    for task in task_list:
        ids, runtime, dependencies, data = task
        for node in dependencies:
            if "M" in node and movement_dictionaries:
                data = datamove_task_meta_info[node][2][0]
                data_info = data_config[data]
                # print("Data: ", data, "Data Info: ", data_info[0])
                weight = str(data_info[0])
                graph.add_edge(node, tuple(ids), weight=weight, label=weight)
            elif "M" not in node:

                if movement_dictionaries:
                    if check_redundant:
                        redundant = False
                        # Make sure it is not a redundant edge
                        targets_data_tasks = compute_tid_to_datamove_tid[tuple(ids)]
                        for data_task in targets_data_tasks:
                            data_task_info = datamove_task_meta_info[data_task]
                            if node in data_task_info[1]:
                                redundant = True
                    else:
                        redundant = False

                    if not redundant:
                        graph.add_edge(node, tuple(ids))
                else:
                    # Weight is the size of all incoming read data
                    weight = 0
                    for data in read_dict[tuple(ids)]:
                        data_info = data_config[data]
                        weight += data_info[0]

                    weight = str(weight)
                    graph.add_edge(node, tuple(
                        ids), weight=weight, label=weight)

    if movement_dictionaries:
        for data_task in data_tasks:
            ids, dependencies, data = data_task
            graph.add_node(ids, data=data)

            for node in dependencies:
                graph.add_edge(node, ids)

    return graph


def get_valid_order(graph):
    """
    Get a valid order of the graph.
    """
    import networkx as nx
    nodes = nx.topological_sort(graph)
    order = [str_to_tuple(str(node)) for node in nodes]
    return order


def get_subgraph(graph, nodes):
    """
    Get a subgraph of the graph.
    """
    import networkx as nx
    subgraph = graph.copy()
    all_nodes = graph.nodes()
    remove_nodes = [node for node in all_nodes if node not in nodes]
    print(remove_nodes)
    subgraph.remove_nodes_from(remove_nodes)
    return subgraph


def make_networkx_datagraph(task_list, task_dictionaries, data_config, movement_dictionaries=None, plot_isolated=False, plot_weights=False):
    import hypernetx as hnx
    import networkx as nx
    runtime_dict, dependency_dict, write_dict, read_dict, count_dict = task_dictionaries

    H = hnx.Hypergraph(read_dict)
    dualH = H.dual()

    # Add weights for data size
    #for e in dualH.edges():
    #    e.weights = data_config[int(str(e))][0]

    H = dualH.dual()

    return H, dualH


def plot_hypergraph(hgraph):
    import hypernetx as hnx
    hnx.draw(hgraph)
    plt.show()


def task_status_color_map(state):
    colors = ["blue", "red", "orange", "purple", "pink"]
    if state.status == "completed":
        return "grey"
    elif state.status == "active":
        # return "red"
        return colors[state.device+1]
    elif state.status == "mapped":
        return "yellow"


def task_device_color_map(state):
    colors = ["blue", "red", "yellow", "purple", "pink"]
    return colors[state.device+1]


def plot_graph(graph, state=None, color_map=None, output="graph.png",
               data_dict=None):
    import networkx as nx

    if data_dict is not None:
        read_dict, write_dict, dependency_dict = data_dict

    if (state is not None) and (color_map is not None):
        for node in graph.nodes():
            graph.nodes[node]['style'] = 'filled'
            graph.nodes[node]['color'] = color_map(state[node])

    if data_dict is not None:
        for node in graph.nodes():
            print("HERE")
            out_edges = graph.out_edges(nbunch=[node])
            print(out_edges)
            for edge in out_edges:
                if ("M" not in edge[1]) and ("M" not in edge[0]):
                    depenendices = dependency_dict[edge[1]]
                    read_list_target = read_dict[edge[1]]
                    read_list_source = read_dict[edge[0]]
                    write_list_target = write_dict[edge[1]]
                    write_list_source = write_dict[edge[0]]

                    flag = 0
                    for read_target in read_list_target:
                        if read_target in write_list_source:
                            print("RED")
                            graph.edges[edge]['color'] = 'red'
                            graph.edges[edge]['style'] = 'dashed'
                            flag = 1
                            break

                    if flag == 1:
                        break

                    for read_target in read_list_target:
                        if read_target in read_list_source:
                            print("GREEN")
                            graph.edges[edge]['color'] = 'green'
                            graph.edges[edge]['style'] = 'dashed'
                            flag = 1
                            break


    pg = nx.drawing.nx_pydot.to_pydot(graph)
    png_str = pg.create_png(prog='dot')
    pg.write_png(output)


def search(list, object):
    for i in range(len(list)):
        if list[i] == object:
            return i
    return None


def search_attribute(list, object, attribute):
    for i in range(len(list)):
        if getattr(list[i], attribute) == object:
            return i
    return None


def search_field(list, object, field, attribute="info"):
    for i in range(len(list)):
        if getattr(list[i], attribute)[field] == object:
            return i
    return None
