import PIL.Image
import PIL.ImageTk
from tkinter import *
from re import S
import matplotlib.pyplot as plt
from fractions import Fraction
import matplotlib.pyplot as plt

from data import PArray

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


class Drainer(object):
    # TODO(hc): better name?
    def __init__(self, q):
        self.q = q

    def __iter__(self):
        while True:
            try:
                if len(self.q) == 0:
                    break
                yield self.q.get()
            except queue.Empty:
                break


def initialize_data(data_config, device_map):
    data_list = dict()
    device_list = device_map.keys()
    n_gpus = len([d for d in device_list if d >= 0])

    # NOTE: Assumes initialization on ONLY one location.
    for data_name, data_info in data_config.items():
        device = device_map[data_info[1]]
        data = PArray("D"+str(data_name),
                      data_info[0], [device])
        device.add_data(data)
        data_list[data.name] = data
    return data_list


def form_device_map(devices):
    device_map = dict()
    for device in devices:
        device_map[device.id] = device
    return device_map


def order_tasks(tasks, order):
    task_list = []
    idx = 0
    for task_name in order:
        task_list.append(tasks[task_name])
        task_list[idx].order = idx
        idx = idx + 1
    return task_list


def plot_active_tasks(device_list, state, type="all"):
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # print(state.state_list)

    times, states = state.unpack()
    active_task_counts = dict()

    for device in device_list:
        active_task_counts[device.name] = []

    for state in states:
        for device in device_list:
            if device.name in state:
                n_tasks = state[device.name].count_active_tasks(type)
                active_task_counts[device.name].append(n_tasks)

    #print(times, active_task_counts)

    for device in device_list:
        plt.step(
            times, active_task_counts[device.name], where="post", label=device.name)
    plt.xlabel("Time (s)")
    plt.ylabel("# active tasks")
    plt.legend()
    plt.title("Active Tasks")
    plt.show()


def plot_memory(device_list, state):
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    times, states = state.unpack()
    memory_count = dict()

    for device in device_list:
        memory_count[device.name] = []

    for state in states:
        for device in device_list:
            if device.name in state:
                print(state)
                print(device.name)
                memory = state[device.name].used_memory()
                memory_count[device.name].append(memory)

    for device in device_list:
        plt.step(times, memory_count[device.name],
                 where="post", label=device.name)
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (bytes)")
    plt.legend()
    plt.title("Used Memory")
    plt.show()


def plot_acus(device_list, state):
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    times, states = state.unpack()
    memory_count = dict()

    for device in device_list:
        memory_count[device.name] = []

    for state in states:
        for device in device_list:
            if device.name in state:
                memory = state[device.name].used_acus()
                memory_count[device.name].append(memory)

    for device in device_list:
        plt.step(times, memory_count[device.name],
                 where="post", label=device.name)

    plt.xlabel("Time (s)")
    plt.ylabel("ACUs")
    plt.legend()
    plt.title("Used ACUs")
    plt.show()


def plot_transfers_data(data, device_list, state, total=False):
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    times, states = state.unpack()
    count = dict()

    for device in device_list:
        count[device.name] = []
    count["total"] = []

    for state in states:
        s = 0
        for device in device_list:
            if device.name in state:
                try:
                    transfers = state[data.name].transfers[device.name]
                except KeyError:
                    transfers = 0

                count[device.name].append(transfers)
                s += transfers
        count["total"].append(s)

    if not total:
        for device in device_list:
            plt.step(times, count[device.name],
                     where="post", label=device.name)
    else:
        plt.step(times, count["total"], where="post", label="total")

    plt.xlabel("Time (s)")
    plt.ylabel("# Transfers")
    plt.legend()
    plt.title("Movement Count")
    plt.show()


def make_image_folder_time(end_time, step, folder_name, state):
    import os
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    time_stamp = 0.0
    idx = 0
    while time_stamp < end_time:
        filename = folder_name + "/" + "graph_" + str(idx) + ".png"
        point = state.get_logs_with_time(time_stamp)
        plot_graph(graph_full, point, task_status_color_map, output=filename)
        time_stamp += step
        idx += 1


def make_image_folder(folder_name, state, increment=True):
    import os
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    inc = 0
    for time_state in state.state_list:
        time_stamp = time_state[0]
        if increment:
            filename = folder_name + "/" + "graph_" + str(inc) + ".png"
        else:
            filename = folder_name + "/" + "graph_" + str(time_stamp) + ".png"
        point = time_state[1]
        plot_graph(graph_full, point, task_status_color_map, output=filename)
        inc += 1


def make_movie(folder_name):
    import cv2
    import glob

    frameSize = (500, 500)
    out = cv2.VideoWriter('output_video.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize)
    for filename in glob.glob(folder_name + "/*.png"):
        img = cv2.imread(filename)
        out.write(img)
    out.release()


def make_interactive(end_time, state):

    root = Tk()
    root.wm_title("Window")
    root.geometry("1000x1000")

    v1 = DoubleVar()

    iterate = False

    def stop():
        nonlocal iterate
        iterate = (not iterate)

    def show(value):
        time = float(value)
        plot_graph(graph_full, state.get_logs_with_time(
            time), task_status_color_map)
        load = PIL.Image.open("graph.png")
        load = load.resize((450, 350))
        render = PIL.ImageTk.PhotoImage(load)
        img = Label(root, image=render)
        img.image = render
        img.place(x=250, y=100)
        sel = str(time)
        l1.config(text=sel)

    s1 = Scale(root, variable=v1, from_=0, to=end_time,
               orient=HORIZONTAL, length=600, resolution=0.01, state='active', command=show)

    def refreshscale():
        nonlocal iterate
        # print("Refresh", iterate)
        if iterate:
            v = s1.get()

            def increment(v):
                if v > end_time:
                    return 0.0
                else:
                    v = v + 0.5
                    return v
            vafter = increment(v)
            s1.set(vafter)
        root.after(500, refreshscale)

    l3 = Label(root, text="Time Slider")
    b1 = Button(root, text="Start/Stop", command=stop)
    l1 = Label(root)

    s1.pack(anchor=CENTER)
    l3.pack()
    b1.pack(anchor=CENTER)
    l1.pack()
    refreshscale()
    root.mainloop()


units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
# Alternative unit definitions, notably used by Windows:
# units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}


def parse_size(size):
    number, unit = [string.strip() for string in size.split()]
    return int(float(number)*units[unit])


def get_trivial_mapping(task_handles):
    import random
    mapping = dict()

    for task_name in task_handles.keys():
        task_handle = task_handles[task_name]
        valid_devices = task_handle.get_valid_devices()
        print(valid_devices)
        valid_devices.sort()
        #mapping[task_name] = valid_devices[0]
        mapping[task_name] = random.choice(valid_devices)

    return mapping


