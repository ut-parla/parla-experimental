import PIL.Image
import PIL.ImageTk
from tkinter import *
from re import S
import numpy as np
import queue
from queue import PriorityQueue, Queue
from fractions import Fraction as F
import matplotlib.pyplot as plt
import time
import warnings
import copy

from collections import namedtuple
from fractions import Fraction
import matplotlib.pyplot as plt
import heapq


class PriorityQ:
    def __init__(self):
        self.queue = []

    def put(self, item):
        heapq.heappush(self.queue, item)

    def get(self):
        return heapq.heappop(self.queue)

    def __len__(self):
        return len(self.queue)


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
    data_task_dict = dict()
    task_to_movement_dict = dict()

    count = n_tasks

    for task in task_list:
        # For each task create all data movement tasks
        ids, runtime, dependencies, data = task
        data_dependencies = task_data_dependencies[tuple(ids)]
        task_to_movement_dict[tuple(ids)] = []

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
            data_task_dict[movement_id] = data_task_tuple
            task_to_movement_dict[tuple(ids)].append(movement_id)
            count += 1
            count_dict[movement_id] = count

    return data_tasks, data_task_dict, task_to_movement_dict


def make_networkx_graph(task_list, task_dictionaries, data_config, movement_dictionaries=None, plot_isolated=False, plot_weights=False, check_redundant=True):
    """
    Create a networkx graph from the task list.
    """
    import networkx as nx
    graph = nx.DiGraph()

    # movement_dictionaries = None
    runtime_dict, dependency_dict, write_dict, read_dict, count_dict = task_dictionaries

    if movement_dictionaries is not None:
        data_tasks, data_task_dict, task_to_movement_dict = movement_dictionaries

    for task in task_list:
        ids, runtime, dependencies, data = task
        # print("Runtime: ", (runtime))
        # print("Data: ", data)
        graph.add_node(tuple(ids), data=data)

    for task in task_list:
        ids, runtime, dependencies, data = task
        for node in dependencies:
            if "M" in node and movement_dictionaries:
                data = data_task_dict[node][2][0]
                data_info = data_config[data]
                # print("Data: ", data, "Data Info: ", data_info[0])
                weight = str(data_info[0])
                graph.add_edge(node, tuple(ids), weight=weight, label=weight)
            elif "M" not in node:

                if movement_dictionaries:
                    if check_redundant:
                        redundant = False
                        # Make sure it is not a redundant edge
                        targets_data_tasks = task_to_movement_dict[tuple(ids)]
                        for data_task in targets_data_tasks:
                            data_task_info = data_task_dict[data_task]
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
            out_edges = graph_full.out_edges(nbunch=[node])
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


class CanAssign(object):
    def __init__(self, q1, q2, pool):
        self.q1 = q1
        self.q2 = q2
        self.pool = pool
        self.state = 0

    def __iter__(self):
        while True:
            try:
                # print("CanAssign: ", self.q.queue)

                if not len(self.q.queue):
                    break
                # Check if top of queue is ready
                # print("Before", self.q.queue[0].name, self.q.queue[0].status)
                self.q.queue[0].update_status()
                # print("After", self.q.queue[0].name, self.q.queue[0].status)

                if self.q.queue[0].status != 1:
                    break

                # Check if top of queue can fit in memory
                if not self.q.queue[0].check_resources(self.pool):
                    break

                # Check if there is enough data to evict to make room for new task
                # TODO implement this
                # if so, yield an eviction task

                yield self.q.get()

            except queue.Empty:
                break


class Assign2(object):
    def __init__(self, q1, q2, pool):
        self.q1 = q1
        self.q2 = q2
        self.pool = pool
        self.state = 0
        self.fail_count = 0

    def create_eviction_task(self, parent_task):
        # TODO implement this
        pass

    def __iter__(self):
        while True:
            try:
                if self.fail_count > 1:
                    # print("Exceeded failure count")
                    self.fail_count = 0
                    yield None

                if self.state == 0:
                    # try to assign data movement
                    self.q = self.q2
                    self.state = 1 - self.state
                elif self.state == 1:
                    # try to assign computation
                    self.q = self.q1
                    self.state = 1 - self.state

                # Is the queue empty
                if not len(self.q.queue):
                    # print("Queue is empty")
                    self.fail_count += 1
                    if self.fail_count > 1:
                        self.fail_count = 0
                        break
                    continue

                # print("Queue Status in Assignment: ", self.q.queue,
                #      self.fail_count, self.state)

                # Is top of queue ready
                self.q.queue[0].update_status()
                if self.q.queue[0].status != 1:
                    # print("Assignment Failed: Top of queue is not ready")
                    self.fail_count += 1
                    if self.fail_count > 1:
                        self.fail_count = 0
                        break
                    continue

                # Does the task fit in memory
                if not self.q.queue[0].check_resources(self.pool):
                    # print("Assignment Failed: Task does not fit in memory")
                    self.fail_count += 1
                    if self.fail_count > 1:
                        self.fail_count = 0
                        break
                    continue

                # If compute task, check if there is enough data to evict to make room for new task
                # TODO implement this
                # if so, create and yield an eviction task

                # print("Success: Return Task")
                self.fail_count = 0
                yield self.q.get()

            except (queue.Empty, StopIteration) as error:
                # print("Queue is empty (error catch)")
                break


class DataStatus:
    def __init__(self, _stale=0, _used=0, _prefetch=0, _in_progress=False):
        self.stale = _stale
        self.used = _used
        self.prefetched = _prefetch
        self.in_progress = _in_progress

        # Tasks that depend on this data (that have been prefetched for)
        self.dependent_tasks = []

        # Tasks that are currently moving this data (e.g. prefetching)
        self.moving_tasks = []  # PriorityQueue()

        # Tasks that are removing this data (e.g. evicting from this device)
        self.eviction_tasks = []

    def __str__(self):
        return f"Data State :: stale: {self.stale} | used: {self.used} | prefetch: {self.prefetched} | in_progress: {self.in_progress}"

    def __repr__(self):
        return f"({self.stale}, {self.used}, {self.prefetched}, {self.in_progress})"

    def __hash__(self):
        return hash((self.stale, self.used, self.prefetched, self.in_progress))


class Data:

    dataspace = dict()

    def __init__(self, _name, _size, device_list):
        self.name = _name
        self.size = _size

        self.locations = dict()

        for device in device_list:
            self.locations[device.name] = DataStatus()

        # Must be set before use.
        Data.dataspace[self.name] = self

        # heuristics for eviction
        self.info = dict()

        self.transfers = dict()

    def __str__(self):
        return f"Data: {self.name} {self.size} | State: {self.locations}"

    def get_status(self, device):
        if device.name in self.locations:
            return self.locations[device.name]
        else:
            return False

    def add_transfer(self, device):
        if device.name in self.transfers:
            self.transfers[device.name] += 1
        else:
            self.transfers[device.name] = 1

    def get_dependent_tasks(self, device_list=None):
        dependent_tasks = []
        if device_list is None:
            for device in self.locations:
                dependent_tasks.extend(
                    self.locations[device.name].dependent_tasks)
        else:
            for device in device_list:
                dependent_tasks.extend(
                    self.locations[device.name].dependent_tasks)
        return dependent_tasks

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def update_state(self, device, stale=0, used=0, prefetch=0):
        if device.name in self.locations:
            self.locations[device.name].stale = stale
            self.locations[device.name].used = used
            self.locations[device.name].prefetched = prefetch
            return True
        return False

    def increment_state(self, device, stale=False, used=False, prefetch=False):
        if device.name in self.locations:
            if stale:
                self.locations[device.name].stale += 1
            if used:
                self.locations[device.name].used += 1
            if prefetch:
                self.locations[device.name].prefetched += 1
            return True
        return False

    def decrement_state(self, device, stale=False, used=False, prefetch=False):
        if device.name in self.locations:
            if stale:
                self.locations[device.name].stale -= 1
            if used:
                self.locations[device.name].used -= 1
            if prefetch:
                self.locations[device.name].prefetched -= 1
            return True
        return False

    def update(self, device, state, value):
        if device.name in self.locations:
            setattr(self.locations[device.name], state, value)
            return True
        return False

    def query(self, device, property):
        if device.name in self.locations:
            if hasattr(self.locations[device.name], property):
                return getattr(self.locations[device.name], property)
        return None

    def valid(self, device, allow_in_progress=False):
        # Does the data have a valid copy on the device?
        if device.name in self.locations:
            if self.locations[device.name].stale:
                return False
            if not allow_in_progress and self.locations[device.name].in_progress:
                return False
        else:
            return False
        return True

    def evict(self, device_name):
        if device_name in self.locations:
            if self.locations[device_name].stale and not self.locations[device_name].used:

                # Delete from data table
                del self.locations[device_name]

                # Delete from device list
                device = SyntheticDevice.devicespace[device_name]
                device.delete_data(self)

            elif self.locations[device_name].stale and self.locations[device_name].used:
                raise Exception("Invalid State. Data is stale and used.")

    def evict_stale(self):
        device_names = list(self.locations.keys())
        state = list(self.locations.values())
        for device_name, state in zip(device_names, state):
            # print("Eviction: ", device_name, state)
            if state.stale:
                self.evict(device_name)

    def set(self, key, value):
        self.info[key] = value

    def decrement(self, key, value=0):
        if key in self.info:
            self.info[key] -= 1
        else:
            self.info[key] = value

    def increment(self, key, value=0):
        if key in self.info:
            self.info[key] += 1
        else:
            self.info[key] = value

    def create_on_device(self, device, in_progress=True):
        if device.name not in self.locations:
            self.locations[device.name] = DataStatus(0, 0, 0, in_progress)

        # if in_progress and not self.locations[device.name].in_progress:
        #    self.locations[device.name].in_progress = True

    def unlock(self, device_list):
        # Decrement used counter
        for device in device_list:
            if device.name in self.locations:
                print("Unlocking data: ", self.name, device_list)
                self.locations[device.name].used -= 1
                assert(self.locations[device.name].used >= 0)
            else:
                raise Exception(
                    "Attempting to unlock data that is not on device.")

    def lock(self, device_list):
        # Increment used counter
        for device in device_list:
            if device.name in self.locations:
                print("Locking data: ", self.name, device_list)
                self.locations[device.name].used += 1
                assert(self.locations[device.name].used >= 0)
            else:
                raise Exception(
                    "Attempting to lock data that is not on device.")

    def start_prefetch(self, calling_task, task_list, device_list):
        for device in device_list:

            # print("OLD: ", self)
            # Add to data table (Create state if not already there)
            self.create_on_device(device, in_progress=True)
            # print("NEW: ", self)

            # Add to device pool
            # NOTE: Must be done after create_on_device (because state is used to update the memory tracking)
            device.add_data(self)

            assert(device.name in self.locations)

            # Increment prefetch counter
            # print("Incrementing prefetch counter", self, task_list)
            self.locations[device.name].prefetched += 1
            self.locations[device.name].dependent_tasks.extend(task_list)

            assert(calling_task.completion_time >= 0)
            # self.locations[device.name].moving_tasks.put(
            #    (calling_task.completion_time, calling_task))
            self.locations[device.name].moving_tasks.append(calling_task)

    def finish_prefetch(self, calling_task, task_list, device_list):
        for device in device_list:

            # if device.name in self.locations:
            assert(device.name in self.locations)
            self.locations[device.name].in_progress = False

            self.locations[device.name].moving_tasks.remove(calling_task)
            # finished_task = self.locations[device.name].moving_tasks.get()[1]
            # assert(finished_task == calling_task)

            # print("FINISH Prefetch Check", self)

    def use(self, task, device_list, is_movement):
        # Decrement prefetch counter
        for device in device_list:

            if device.name in self.locations and not is_movement:
                self.locations[device.name].prefetched -= 1
                # print("USEING")
                # print(self.locations[device.name].dependent_tasks)
                # print(task.name)
                self.locations[device.name].dependent_tasks.remove(task)
            elif not is_movement:
                raise Exception(
                    f"Invalid State. Data is not available by task runtime. \n\t {task} | \n\t {data} | \n\t {device}")

    def read(self, task, device_list, is_movement=False):
        # Decrement prefetch counter
        self.use(task, device_list, is_movement)

    def write(self, task, device_list, is_movement=False):

        # NOTE: Write data isn't prefetched.
        # Decrease prefetch count
        # self.use(task, device_list, is_movement)

        #Check if data is already being used on device.

        if not is_movement:

            for device in device_list:
                if(self.locations[device.name].used > 1):
                    print(self.name, self.locations)
                    assert(False)

            for device_name in self.locations.keys():
                if SyntheticDevice.devicespace[device_name] not in device_list:
                    if(self.locations[device_name].used > 0):
                        print(self.name, self.locations)
                        assert(False)

            # Mark stale
            device_name_list = [d.name for d in device_list]
            for device_name in self.locations.keys():
                if device_name not in device_name_list:
                    self.locations[device_name].stale += 1

            # Clear stale data from devices
            # NOTE: Should be no-op if is_movement is True
            self.evict_stale()

    def active_transfer(self, device):
        # Is the data currently being moved/created?
        if status := self.get_status(device):
            return status.in_progress
        else:
            return False

    def active_use(self, device):
        # Is the data currently used by a task
        if status := self.get_status(device):
            return status.used > 0
        return False

    def active_need(self, device):
        # Is the data on device already prefetched and needed for any scheduled tasks?
        if status := self.get_status(device):
            return status.prefetched > 0
        return False

    def get_valid_sources(self, allow_in_progress=False):
        valid_sources = []
        for device_name in self.locations.keys():
            device = SyntheticDevice.devicespace[device_name]
            if self.valid(device, allow_in_progress=allow_in_progress):
                valid_sources.append(device)
        return valid_sources

    def choose_source(self, target_device, pool, required=False):
        topology = pool.topology

        # if already in progress at target, return that as the source
        # print("CHECK SOURCES IN PROGRESS :: ", self, target_device)
        if self.active_transfer(target_device):
            return target_device

        # otherwise, find the best source

        # Get list of non-stale devices that hold this data
        source_list = self.get_valid_sources()
        # print("SOURCE LIST", source_list)

        # Find the closest free source for the data
        closest_free_source = topology.find_best(
            target_device, source_list, free=True)

        return closest_free_source

    # def __del__(self):
    #    del Data.dataspace[self.name]



class BandwidthHandle(object):

    def make_data(source, size):
        if source.idx >=0:
            with cupy.cuda.Device(source.idx) as device:
                data = cp.ones(size, dtype=cp.float32)
                device.synchronize()
        else:
            data = np.ones(size, dtype=np.float32)
        return data

    def copy(self, arr, source, destination):
        #Assume device.idx >= 0 means the device is a GPU.
        #Assume device.idx < 0 means the device is a CPU.
        if source.idx >= 0 and destination.idx < 0:
            #Copy GPU to CPU
            with cp.cuda.Deice(destination.idx):
                with cp.cuda.Stream(non_blocking=True) as stream:
                    membuffer = cp.asnumpy(arr, stream=stream)
                    stream.synchronize()
            return membuffer
        elif source.idx < 0 and destination.idx >= 0:
            #Copy CPU to GPU
            with cp.cuda.Device(destination.idx):
                with cupy.cuda.Stream(non_blocking=True) as stream:
                    membuffer = cp.empty(arr.shape, dtype=arr.dtype)
                    membuffer.set(arr, stream=stream)
                    stream.synchronize()
            return membuffer
        elif source.idx >= 0 and destination.idx >= 0:
            #Copy GPU to GPU
            with cp.cuda.Device(destination.idx):
                with cupy.cuda.Stream(non_blocking=True) as stream:
                    membuffer = cp.empty(arr.shape, dtype=arr.dtype)
                    membuffer.data.copy_from_device_async(array.data, array.nbytes, stream=stream)
                    stream.synchronize()
            return membuffer
        elif source.idx < 0 and destination.idx < 0:
            #Copy CPU to CPU
            return np.copy(arr)
        else:
            raise Exception("I'm not sure how you got here. But we don't support this device combination")

    @staticmethod
    def estimate(self, source, destination, size=10**6, samples=20):
        times= []
        for i in range(samples):
            array = self.make_data(source, size)
            start = time.perf_counter()
            self.copy(array, source, destination)
            end = time.perf_counter()
            times.append(end-start)

        times = np.asarray(times)
        return np.means(times)


class SyntheticTopology(object):

    def __init__(self, name, device_list, bandwidth=None, connections=None):
        self.name = name
        self.devices = device_list

        # Find CPU (assume unique, use first found)
        for device in self.devices:
            if device.id < 0:
                self.host = device
                break

        # Id -> Device Mapping
        self.id_map = {device.id: device for device in self.devices}

        if bandwidth is None:
            bandwidth = np.zeros((len(self.devices), len(self.devices)))
        self.bandwidth = bandwidth

        if connections is None:
            connections = np.zeros(
                (len(self.devices), len(self.devices)), dtype=np.int32)

        self.connections = connections

        self.active_connections = np.zeros(
            (len(self.devices), len(self.devices)), dtype=np.int32)

        # Set backlink to devices (just in case we ever need to query it)
        for device in self.devices:
            device.topology = self
            # All devices are connected to the host
            self.add_connection(self.host, device, symmetric=True)

            # All devices are connected to themselves
            self.add_connection(device, device)

        self.max_copy = dict()
        self.active_copy = dict()
        for d in device_list:
            self.max_copy[d.name] = d.copy_engines
            self.active_copy[d.name] = 0

    def __str__(self):
        return f"Topology: {self.name} | \n\t Devices: {self.devices} | \n\t Bandwidth: {self.bandwidth} | \n\t Connections: {self.connections}"

    def __repr__(self):
        return str(self.name)

    def __hash__(self):
        return hash(self.name)

    def add_bandwidth(self, device1, device2, value, reverse=None):
        self.bandwidth[device1.idx, device2.idx] = value

        if reverse is not None:
            self.bandwidth[device2.idx, device1.idx] = reverse

    def get_device(self, idx):
        return self.id_map[idx]

    def sample_bandwidth(self, device1, device2, size=10**6, samples=20):
        self.bandwidth[device1.idx, device2.idx] = BandwidthHandle.estimate(
            device1, device2, size, samples)
        self.bandwidth[device2.idx, device1.idx] = BandwidthHandle.estimate(
            device2, device1, size, samples)

    def fill_bandwidth(self, size=10**6, samples=20):
        for device1 in self.devices:
            for device2 in self.devices:
                self.sample_bandwidth(device1, device2, size, samples)

    def add_connection(self, device1, device2, symmetric=True):
        if symmetric:
            self.connections[device1.idx, device2.idx] = 1
            self.connections[device2.idx, device1.idx] = 1
        else:
            self.connections[device1.idx, device2.idx] = 1

    def remove_connection(self, device1, device2, symmetric=True):
        if symmetric:
            self.connections[device1.idx, device2.idx] = 0
            self.connections[device2.idx, device1.idx] = 0
        else:
            self.connections[device1.idx, device2.idx] = 0

    def add_usage(self, device1, device2):
        self.active_connections[device1.idx, device2.idx] += 1
        # self.active_connections[device2][device1] += 1

        if device1 == device2:
            return

        self.active_copy[device1.name] += 1
        self.active_copy[device2.name] += 1

        if self.connections[device1.idx, device2.idx] <= 0:
            # Assume communication is through CPU buffer
            self.active_connections[device1.idx, self.host.idx] += 1
            self.active_connections[self.host.idx, device2.idx] += 1
            self.active_copy[self.host.name] += 1

    def decrease_usage(self, device1, device2):
        self.active_connections[device1.idx, device2.idx] -= 1
        # self.active_connections[device2][device1] -= 1

        if device1 == device2:
            return

        self.active_copy[device1.name] -= 1
        self.active_copy[device2.name] -= 1

        if self.connections[device1.idx][device2.idx] <= 0:
            # Assume communication is through CPU buffer
            self.active_connections[device1.idx, self.host.idx] -= 1
            self.active_connections[self.host.idx, device2.idx] -= 1
            self.active_copy[self.host.name] -= 1

    def check_free(self, device1, device2, symmetry=True, engines=True):

        if device1 == device2:
            return True

        if engines:
            if self.active_copy[device1.name] >= self.max_copy[device1.name]:
                return False
            if self.active_copy[device2.name] >= self.max_copy[device2.name]:
                return False

        # TODO: Will need to be adjusted for connections that can support more than 1 simultaneous copy

        if symmetry:
            return self.active_connections[device1.idx, device2.idx] == 0 and self.active_connections[device2.idx, device1.idx] == 0
        else:
            return self.active_connections[device1.idx, device2.idx] == 0

    def find_best(self, target, source_list, free=True):

        if free:
            # Find the closest free source for the data
            closest_free_source = None
            closest_free_distance = np.inf
            for source in source_list:
                if self.check_free(target, source, symmetry=True, engines=True):
                    distance = self.bandwidth[target.idx, source.idx]

                    if source == target:
                        distance = 0

                    if distance < closest_free_distance:
                        closest_free_source = source
                        closest_free_distance = distance
            return closest_free_source
        else:
            # Find the closest source for the data
            closest_source = None
            closest_distance = np.inf
            for source in source_list:
                distance = self.bandwidth[target.idx, source.idx]

                if source == target:
                    distance = 0

                if distance < closest_distance:
                    closest_source = source
                    closest_distance = distance
            return closest_source

    def compute_transfer_time(self, current_time, targets, data_sources):
        transfer_time = current_time
        for data_name, source in data_sources.items():
            target = targets[data_name]
            data = Data.dataspace[data_name]

            # Is the data already being transfer to this device?
            # In this case, we have to wait for the completition of the transfer
            if data.active_transfer(target):
                if status := data.get_status(target):
                    earliest_finish = status.moving_tasks[0].completion_time
                transfer_time = max(transfer_time, earliest_finish)

            # Otherwise, compute the expected time with the bandwidth
            if source == target:
                # No need to transfer data
                transfer_time += 0
            else:
                transfer_time += data.size / \
                    self.bandwidth[source.idx, target.idx]

        return transfer_time

    def reserve_communication(self, data_targets, data_sources):

        # Reserve the communication links for the data
        reserved = dict()

        for data_name in data_targets.keys():
            target_device = data_targets[data_name]
            source_device = data_sources[data_name]
            data = Data.dataspace[data_name]

            # Only take resources if the transfer is not already active
            if (not data.active_transfer(target_device)) and (not (target_device == source_device)):
                reserved[data_name] = True
                self.add_usage(target_device, source_device)
                data.add_transfer(target_device)

            return reserved

    def free_communication(self, data_targets, data_sources, reserved):
        for data_name in data_targets.keys():
            target_device = data_targets[data_name]
            source_device = data_sources[data_name]
            data = Data.dataspace[data_name]

            # Only free resources if the transfer is not already active
            if data_name in reserved:
                self.decrease_usage(target_device, source_device)


class ResourcePool:

    def __init__(self):
        self.pool = dict()

    def __str__(self):
        return f"ResourcePool: {self.pool}"

    def __repr__(self):
        return f"ResourcePool: {self.pool}"

    def __hash__(self):
        return hash(self.name)

    def add_resource(self, device, resource, amount):
        if device not in self.pool:
            self.pool[device] = dict()
        self.pool[device][resource] = amount

    def add_resources(self, device, resources):
        for resource, amount in resources.items():
            self.add_resource(device, resource, amount)
        return True

    def set_toplogy(self, topology):
        self.topology = topology

        for device in self.topology.devices:
            self.add_resources(device.name, device.resources)

            # Add backlink to device (in case we ever need to query it there)
            device.active_memory = self

    def finalize(self):
        self.max_pool = copy.deepcopy(self.pool)
        self.max_connections = copy.deepcopy(self.connections)

    def check_device_resources(self, device, requested_resources):
        # print(device, requested_resources, self.pool)
        if device.name not in self.pool:
            return False
        for resource, amount in requested_resources.items():
            if resource not in self.pool[device.name]:
                return False
            if resource == "memory":
                # print("Checking memory")
                # print(self.pool[device.name][resource],
                #      device.persistent_memory, amount)
                if (self.pool[device.name][resource] - device.persistent_memory) < amount:
                    return False
            else:
                if self.pool[device.name][resource] < amount:
                    return False
        return True

    def check_resources(self, requested_resources):
        for device_name, resources in requested_resources.items():
            device = SyntheticDevice.devicespace[device_name]
            if not self.check_device_resources(device, resources):
                return False
        return True

    def reserve_resources_device(self, device, requested_resources):
        for resource, amount in requested_resources.items():
            self.pool[device][resource] -= amount
        return True

    def free_resources_device(self, device, requested_resources):
        for resource, amount in requested_resources.items():
            self.pool[device][resource] += amount
        return True

    def reserve_resources(self, requested_resources):
        for device, resources in requested_resources.items():
            self.reserve_resources_device(device, resources)
        return True

    def free_resources(self, requested_resources):
        for device, resources in requested_resources.items():
            self.free_resources_device(device, resources)
        return True

    def delete_device(self, device):
        del self.pool[device]
        return True

    def delete_resource_device(self, device, resource):
        if device in self.pool:
            if resource in self.pool[device]:
                del self.pool[device][resource]
        return True

    def delete_resources_device(self, device, resources):
        for resource in resources:
            self.delete_resource(self, device, resource)
        return True

    def delete_resource(self, resource):
        for device in self.pool.keys():
            self.delete_resource_device(device, resource)
        return True

    def delete_resources(self, resources):
        for device in self.pool.keys():
            self.delete_resources_device(device, resources)
        return True


class EvictionPolicy:

    def apply(self, datapool, active_tasks, planned_tasks):
        # Assign or update priority to every item in datapool
        pass


class DataPool:

    def __init__(self, _device, _policy):
        self.pool = list()
        self.device = _device
        self.policy = _policy

        self.total_memory = 0
        self.in_progress_memory = 0

    def __getitem__(self, index):
        return self.pool[index]

    def __setitem__(self, index, value):
        # NOTE: DON'T EVER USE THIS. For debugging convience only. Use add_data to append instead.
        self.pool[index] = value

    def __str__(self):
        return f"{[ d.name+repr(d.get_status(self.device)) for d in self.pool]}"

    def __repr__(self):
        return f"{[repr(x) for x in self.pool]}"

    def add_data(self, data):

        # check if data is already in pool
        if data in self.pool:
            return

        # if space on device is available, add to list and consume resources
        self.pool.append(data)

        self.update_memory()

        # Rank by priority
        # self.policy.apply(self.pool, None, None)
        # self.pool.sort(key=lambda x: getattr(
        #    x.locations[self.device], "priority"))

    def evict(self):
        return self.pool.pop()

    def __contains__(self, item):
        return item in self.pool

    def verify(self, data):
        if search_attribute(self.pool, data, "name") is not None:
            return True

    def delete_data(self, data):
        self.pool.remove(data)
        self.update_memory()

    def default_compare(self, val):
        return val > 0

    def get_status(self, data_name):
        return search_attribute(self.pool, data_name, "name").get_status(self.device)

    def extract_list(self, property, compare=None):
        if compare is None:
            compare = self.default_compare

        if property == "all":
            return self.pool
        else:
            sublist = list()
            for data in self.pool:
                if compare(getattr(data, property)):
                    sublist.append(data)
            return sublist

    def extract_list_status(self, property, compare=None):
        if compare is None:
            compare = self.default_compare

        if property == "all":
            return self.pool
        else:
            sublist = list()
            for data in self.pool:
                if compare(getattr(data.locations[self.device.name], property)):
                    sublist.append(data)
            return sublist

    def __get_sublist_size(self, sublist):
        return sum([data.size for data in sublist])

    def find_memory(self, property, compare=None):
        if compare is None:
            compare = self.default_compare

        return self.__get_sublist_size(self.extract_list(property, compare))

    def find_memory_status(self, property, compare=None):
        if compare is None:
            compare = self.default_compare

        return self.__get_sublist_size(self.extract_list_status(property, compare))

    def update_memory(self):
        self.total_memory = self.find_memory_status("all")
        self.in_progress_memory = self.find_memory_status("in_progress")

        self.device.update_persistent_memory()

    def evictable_memory(self, include_prefetched=True):
        size = 0
        for data in self.pool:
            if data.locations[self.device].prefetched and not include_prefetched:
                continue
            if data.locations[self.device.name].used:
                continue
            size += data.size
        return size


class SyntheticDevice:

    count = 0
    devicespace = dict()

    def __init__(self, _name, _id, _resources, copy_engines=2, idx=None, policy=EvictionPolicy()):
        self.name = _name
        self.id = _id
        if idx is not None:
            self.idx = idx
        else:
            self.idx = SyntheticDevice.count
        SyntheticDevice.count += 1

        self.resources = _resources
        self.active_tasks = PriorityQ()
        self.planned_compute_tasks = PriorityQ()
        self.planned_movement_tasks = PriorityQ()
        self.copy_engines = copy_engines

        SyntheticDevice.devicespace[self.name] = self

        self.active_data = DataPool(self, policy)
        self.persistent_memory = 0

    def is_cpu(self):
        return self.id < 0

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        if self.active_memory:
            return f"Device {self.name} | \n\t Planned Data Tasks: {self.planned_movement_tasks.queue} | \n\t Planned Compute Tasks: {self.planned_compute_tasks.queue} | \n\t Active Tasks: {self.active_tasks.queue} | \n\t Resources: {self.active_memory.pool[self.name]} | \n\t Memory: {self.available_memory()} | \n\t Active Data: {self.active_data}"
        else:
            return f"Device {self.name} | \n\t Planned Tasks: {self.planned_tasks.queue} | \n\t Active Tasks: {self.active_tasks.queue} | \n\t Resources: {self.resources}"

    def get_current_resources(self):
        current_resources = copy.deepcopy(self.active_memory.pool[self.name])
        current_resources["memory"] = self.available_memory()
        return current_resources

    def delete_data(self, data):
        return self.active_data.delete_data(data)

    def __hash__(self):
        return hash(self.name)

    def add_data(self, data):
        return self.active_data.add_data(data)

    def available_memory(self, usage=False):

        if not usage:
            if hasattr(self, "active_memory") and self.active_memory is not None:
                if self.name in self.active_memory.pool:
                    return self.active_memory.pool[self.name]["memory"] - self.persistent_memory
            else:
                return False
        else:
            if hasattr(self, "active_memory") and self.active_memory is not None:
                if self.name in self.active_memory.pool:
                    return self.resources["memory"] - (self.active_memory.pool[self.name]["memory"] - self.persistent_memory)
            else:
                return False

    def available_acus(self, usage=False):
        if not usage:
            if self.name in self.active_memory.pool:
                return self.active_memory.pool[self.name]["acus"]
        else:
            if self.name in self.active_memory.pool:
                return 1 - self.active_memory.pool[self.name]["acus"]

    def update_persistent_memory(self):
        self.persistent_memory = self.active_data.total_memory

        # There should always be enough memory to fit both the running tasks AND the persistent data.
        if hasattr(self, "active_memory") and self.active_memory is not None:
            if self.name in self.active_memory.pool:
                assert(self.persistent_memory <
                       self.active_memory.pool[self.name]["memory"])

    def get_nonlocal_size(self, data):
        # Return the size of data that needs to be moved if not already on the device
        if data in self.active_data:
            return 0

        return data.size

    def check_fit(self, data):
        # Check if the data is already on the device
        if data in self.active_data:
            # TODO: Check if its not stale, i dunno. I don't think theres a good reason to rerequest memory.
            # print("ITS ALREADY THERE (possibly in progress")
            return True
        if available_memory := self.available_memory():
            if data.size <= available_memory:
                return True
        return False

    def check_fit_with_eviction(self, data):
        if self.check_fit(data):
            return True

        if available_memory := self.available_memory():
            if evictable_memory := self.evictable_memory():
                if data.size <= available_memory + evictable_memory:
                    return True
        return False

    def evictable_memory(self, include_prefetched=True):
        return self.active_data.evictable_memory(include_prefetched)

    def resident_memory(self, in_progress=True):
        if in_progress:
            return self.active_data.total_memory - self.active_data.in_progress_memory
        else:
            return self.active_data.total_memory

    def get_data(self, type="all"):
        # options are "in_progress", "prefetched", "all", "used"
        return self.active_data.extract_list(type)

    def evict_best(self):
        return self.active_data.evict()

    def set_concurrent_copies(self, copy_engines):
        self.copy_engines = copy_engines

    def count_active_tasks(self, type):
        if type == "all":
            return len(self.active_tasks.queue)
        elif type == "all_useful":
            return len([task for task in self.active_tasks.queue if task[1].is_redundant == False])
        elif type == "compute":
            return len([task for task in self.active_tasks.queue if task[1].is_movement == False])
        elif type == "movement":
            return len([task for task in self.active_tasks.queue if task[1].is_movement == True])
        elif type == "copy":
            return len([task for task in self.active_tasks.queue if (task[1].is_movement and not task[1].is_redundant)])

    def add_planned_task(self, _task):
        if _task.is_movement:
            self.planned_movement_tasks.put(_task)
        else:
            self.planned_compute_tasks.put(_task)
        return True

    def pop_planned_task(self):
        if self.planned_tasks.queue[0].status == 1:
            return self.planned_tasks.get()
        return False

    def add_local_active_task(self, task):
        self.active_tasks.put((task.completion_time, task))
        return True

    def pop_local_active_task(self):
        return self.active_tasks.get()

    def initialize_resource_tracking(self, resource_pool, EvictionPolicy, data_list=None):
        self.resource_pool = resource_pool
        self.data_pool = DataPool(
            self.name, EvictionPolicy, self.resource_pool)

        if data_list is not None:
            for data in data_list:
                self.data_pool.add_data(data)

    # def __del__(self):
    #    del SyntheticDevice.devicespace[self.name]


def form_device_map(devices):
    device_map = dict()
    for device in devices:
        device_map[device.id] = device
    return device_map


def initialize_data(data_config, device_map):
    data_list = dict()
    device_list = device_map.keys()
    n_gpus = len([d for d in device_list if d >= 0])

    # NOTE: Assumes initialization on ONLY one location.
    for data_name, data_info in data_config.items():
        device = device_map[data_info[1]]
        data = Data("D"+str(data_name),
                    data_info[0], [device])
        device.add_data(data)
        data_list[data.name] = data
    return data_list


class State:

    def __init__(self, graph, level=1):
        state_dict = dict()

        State = namedtuple("State", "status device start_time completion_time")
        for task in graph.nodes():
            state_dict[task] = State(0, None, None, None)

        self.active_state = state_dict
        self.state_list = []
        self.state_list.append((0.0, state_dict))

        self.level = level

    def log_device(self, device):
        if self.level == 1:
            self.active_state[device.name] = copy.deepcopy(device)
        else:
            self.active_state[device.name] = device.get_current_resources()

    def log_data(self, data):
        if self.level == 1:
            self.active_state[data.name] = copy.deepcopy(data)
        else:
            pass
            #self.active_state[data.name] = copy.deepcopy(data.locations)

    def set_task_status(self, task, type, value):
        self.active_state[task] = self.active_state[task]._replace(
            **{type: value})

    def get_task_status(self, task, type):
        return getattr(self.active_state[task], type)

    def advance_time(self, current_time):
        self.state_list.append((current_time, self.active_state))
        self.active_state = copy.deepcopy(self.active_state)

    def get_state_at_time(self, time):
        import bisect
        times, states = zip(*self.state_list)
        idx = bisect.bisect_left(times, time)
        return states[idx]

    def __getitem__(self, idx):
        return self.state_list[idx]

    def unpack(self):
        return zip(*self.state_list)


class SyntheticTask:

    taskspace = dict()

    def __init__(self, _name, _resources, _duration, _dependencies, _dependents, _read_data, _write_data, _data_targets, _order=None):
        self.name = _name
        self.resources = _resources
        self.duration = _duration

        # What tasks does this task depend on?
        self.dependencies = _dependencies

        # What tasks depend on this task?
        self.dependents = _dependents

        # Is this task ready?
        self.status = 0

        # Priority order (if any)
        self.order = _order

        self.completion_time = -1.0

        # Set of data to read and write
        self.write_data = _write_data
        self.read_data = _read_data

        # Where does the data need to be located at task runtime? (Needed for multidevice tasks)
        self.data_targets = _data_targets
        self.data_sources = None

        self.locations = list(self.resources.keys())

        self.is_movement = False

        self.unlock_flag = True
        self.evict_flag = True
        self.free_flag = True
        self.copy_flag = False

        self.is_redundant = False

        SyntheticTask.taskspace[self.name] = self

    def __str__(self):
        return f"Task: {self.name} | \n\t Dependencies: {self.dependencies} | \n\t Resources: {self.resources} | \n\t Duration: {self.duration} | \n\t Data: {self.read_data} : {self.write_data} | \n\t Needed Configuration: {self.data_targets}"

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)

    def __lt__(self, other):
        # If task is running on device, use completion time order
        if self.completion_time > 0 and other.completion_time > 0:
            return self.completion_time < other.completion_time
        # Otherwise use priority order (user assigned)
        else:
            return self.order < other.order

    def __eq__(self, other):
        # Two tasks are equal if they have the same name
        return self.name == other.name

    def make_data_movement(self):
        self.is_movement = True
        self.unlock_flag = False
        self.free_flag = False
        self.copy_flag = True

    def make_evict(self):
        self.evict_flag = False

    def add_dependent(self, dependent):
        self.dependents.append(dependent)

        for dependent in self.dependents:
            dependent.update_status()

    def add_dependency(self, dependency):
        self.dependencies.append(dependency)
        self.status = 0

    def convert_references(self, taskspace):
        self.dependencies = [taskspace[dependency]
                             for dependency in self.dependencies]
        self.dependents = [taskspace[dependent]
                           for dependent in self.dependents]

    def update_status(self):
        self.status = 0
        # print("Updating Status of",  self.name, self.dependencies)
        for dependency in self.dependencies:
            if dependency.status != 2:
                self.status = 0
                return
        self.status = 1

    def check_status(self):
        return self.status

    def start(self):

        # If task is moving data, then update corresponding data blocks and device data pools
        if self.is_movement:
            assert(self.data_sources is not None)
            for data in self.read_data:
                data.start_prefetch(self,
                                    self.dependents, [self.data_targets[data.name]])
                # print("Prefetch Check: ", data)

        # Lock all used data and their sources (if any)
        self.lock_data()

        # Perform read on used data (decrements prefetch counter if not movement)
        for data in self.read_data:
            target = self.data_targets[data.name]
            data.read(self, [target], self.is_movement)

        # Perform write on used data:
        # 1. (causes eviction in the data pool for all others)
        # 2. (decrements prefetch counter)
        # NOTE: Data movement tasks should NOT be writing data, list should be empty.
        for data in self.write_data:
            target = self.data_targets[data.name]
            data.write(self, [target], self.is_movement)

    def finish(self):

        # Update status to completed
        self.status = 2

        # Propogate status to dependents
        for dependent in self.dependents:
            dependent.update_status()

        # Unlock all used data and their sources
        self.unlock_data()

        # If task has moved data, then update corresponding data blocks and device data pools
        if self.is_movement:
            for data in self.read_data:
                data.finish_prefetch(self, self.dependents, [
                    self.data_targets[data.name]])

            # NOTE: Data movement tasks should NOT have any write data
            # for data in self.write_data:
            #    data.prefetch(self.dependents, [self.data_sources[data.name]])

    def find_source(self, data, pool, required=False):
        success_flag = True

        # Find source
        target_device = self.data_targets[data.name]
        source = data.choose_source(target_device, pool, required=required)
        # print("The best free connection is: ", source.name)
        self.data_sources[data.name] = source

        if source is None:
            success_flag = False

        return success_flag

    def assign_data_sources(self, pool, required=False):
        self.data_sources = dict()
        success_flag = True

        for data in self.read_data:
            if not self.find_source(data, pool, required=required):
                success_flag = False

        for data in self.write_data:
            if not self.find_source(data, pool, required=required):
                success_flag = False

        return success_flag

    def compute_non_resident_memory(self):
        sizes = []

        for data in self.read_data:
            device_target = self.data_targets[data]
            size = device_target.get_nonlocal_size(data)
            sizes.append(size)

        for data in self.write_data:
            device_target = self.data_targets[data]
            size = device_target.get_nonlocal_size(data)
            sizes.append(size)

        return sum(sizes)

    def check_data_locations(self):
        # Check if all data is located at the correct location to run the task
        success_flag = True

        for data_name, device in self.data_targets.items():
            data = Data.dataspace[data_name]

            if not data.valid(device):
                success_flag = False
                break

        return success_flag

    def check_resources(self, pool):
        # Check if all resources and communication links are available to run the task
        # Check if all data is located at the correct location to run the task (if not movement)
        success_flag = True

        # Make sure all data transfers can fit on the device
        if self.is_movement:

            for data in self.read_data:
                device_target = self.data_targets[data.name]
                success_flag = success_flag and device_target.check_fit(data)

            # NOTE: This should be empty
            for data in self.write_data:
                device_target = self.data_targets[data.name]
                success_flag = success_flag and device_target.check_fit(data)

        if not success_flag:
            return False

        # Make sure data transfer can be scheduled right now
        if self.is_movement:
            # Try to assign all data movement to active links
            success_flag = success_flag and self.assign_data_sources(
                pool, required=True)
            # print("Data Sources", self.data_sources)
        else:
            # Make sure all data is already loaded
            success_flag = success_flag and self.check_data_locations()

            if not success_flag:
                raise Exception(
                    f"Data missing on device before task start. Task: {self.name}")

        # Make sure enough resources on the device are available to run the task
        if success_flag:
            return pool.check_resources(self.resources)
        else:
            return False

    def reserve_resources(self, pool):
        # Reserve resources and communication links for the task
        pool.reserve_resources(self.resources)

        if self.data_sources:
            self.reserved = pool.topology.reserve_communication(
                self.data_targets, self.data_sources)

            if (len(self.reserved) == 0) and self.is_movement:
                self.is_redundant = True

    def free_resources(self, pool):
        # Free resources and communication links for the task
        if not self.is_movement:
            pool.free_resources(self.resources)

        if self.data_sources:
            pool.topology.free_communication(
                self.data_targets, self.data_sources, self.reserved)

    def estimate_time(self, current_time, pool):
        # TODO: For data movement and eviction, compute from data movement time

        transfer_time = 0
        if self.is_movement:
            transfer_time = pool.topology.compute_transfer_time(current_time, self.data_targets,
                                                                self.data_sources)

        # print(self.name, "Transfer Time:", transfer_time - current_time)

        # TODO: This assumes compute and data movement tasks are disjoint
        if self.is_movement:
            self.completion_time = transfer_time
        else:
            self.completion_time = current_time + self.duration

        assert(self.completion_time >= 0)
        assert(self.completion_time < float(np.inf))

        return self.completion_time

    def lock_data(self):
        # print("Locking Data for Task:", self.name)

        joint_data = (set(self.read_data) | set(self.write_data))
        for data in joint_data:
            to_lock = []

            device_target = self.data_targets[data.name]
            to_lock.append(device_target)

            if self.data_sources:
                device_source = self.data_sources[data.name]
                assert(device_source is not None)
                if not (device_target == device_source):
                    to_lock.append(device_source)

            data.lock(to_lock)


    def unlock_data(self):

        joint_data = (set(self.read_data) | set(self.write_data))

        for data in joint_data:
            to_unlock = []

            device_target = self.data_targets[data.name]
            to_unlock.append(device_target)

            if self.data_sources:
                device_source = self.data_sources[data.name]
                assert(device_source is not None)
                if not (device_source == device_target):
                    to_unlock.append(device_source)

            data.unlock(to_unlock)

        """
        for data in self.write_data:

            to_unlock = []

            device_target = self.data_targets[data.name]
            to_unlock.append(device_target)

            if self.data_sources:
                device_source = self.data_sources[data.name]
                assert(device_source is not None)
                if not (device_source == device_target):
                    to_unlock.append(device_source)

            data.unlock(to_unlock)
        """

    # def __del__(self):
    #    del SyntheticTask.taskspace[self.name]


class TaskHandle:

    data = {}
    device_map = {}
    movement_dict = {}

    def __init__(self, task_id, runtime, dependency, dependants, read, write, associated_movement):
        self.task_id = task_id
        self.runtime_info = runtime
        self.dependency = dependency
        self.dependants = dependants
        self.read = read
        self.write = write
        self.associated_movement = associated_movement

    @ staticmethod
    def set_data(data):
        TaskHandle.data = data

    @ staticmethod
    def set_devices(device_map):
        TaskHandle.device_map = device_map

    @ staticmethod
    def set_movement_dict(movement_dict):
        TaskHandle.movement_dict = movement_dict

    def get_valid_devices(self):
        contraint_set = list(self.runtime_info.keys())
        device_list = list(TaskHandle.device_map.keys())
        n_gpus = len([d for d in device_list if d >= 0])

        all_idx, all_devices = zip(*TaskHandle.device_map.items())

        # Add all devices
        valid_devices = set()
        for idx in contraint_set:
            if idx < 0:
                valid_devices.add(TaskHandle.device_map[idx])
            if idx > 0:
                valid_devices.add(TaskHandle.device_map[(idx-1)])

        # Add all GPU devices
        if 0 in contraint_set:
            for device in all_devices:
                if not device.is_cpu():
                    valid_devices.add(device)

        return list(valid_devices)

    def get_info_on_device(self, device):
        #print("CHECK", self.runtime_info)

        shifted_idx = device.id + 1
        if device.id < 0:
            if device.id in self.runtime_info:
                return self.runtime_info[device.id]

        if shifted_idx > 0:
            if shifted_idx in self.runtime_info:
                return self.runtime_info[shifted_idx]

        if 0 in self.runtime_info:
            return self.runtime_info[0]

        return None

    def make_task(self, device):

        # def __init__(self, _name, _resources, _duration, _dependencies, _dependents, _read_data, _write_data, _data_targets, _order=None)
        name = self.task_id

        info = self.get_info_on_device(device)

        compute_time = info[0]
        acus = info[1]
        gil_count = info[2]
        gil_time = info[3]
        memory = info[4]

        task_time = compute_time + gil_time*gil_count

        resources = {device.name: {"memory": memory, "acus": acus}}
        duration = task_time

        dependencies = self.dependency
        dependants = self.dependants

        read_data = [TaskHandle.data["D"+str(d)] for d in self.read]
        write_data = [TaskHandle.data["D"+str(d)] for d in self.write]

        data_targets = {"D"+str(d): device for d in self.read}

        return SyntheticTask(name, resources, duration, dependencies, dependants, read_data, write_data, data_targets)

    def make_movement_tasks(self, device):

        task_list = []

        for movement in self.associated_movement:
            movement_tuple = TaskHandle.movement_dict[movement]
            name = movement

            resources = {device.name: {"memory": 0, "acus": Fraction(0)}}
            duration = 0

            dependencies = movement_tuple[1]
            dependants = [self.task_id]

            reads = movement_tuple[2]

            read_data = [TaskHandle.data["D"+str(d)]for d in reads]
            data_targets = {"D"+str(d): device for d in reads}

            movement_task = SyntheticTask(
                name, resources, duration, dependencies, dependants, read_data, [], data_targets)
            movement_task.make_data_movement()

            task_list.append(movement_task)

        return task_list


class SyntheticSchedule:

    def __init__(self, _name=None, topology=None):
        if _name is not None:
            self.name = _name
        else:
            self.name = id(self)
        self.time = 0.0

        self.taskspace = dict()
        self.devicespace = dict()
        self.dataspace = dict()

        self.all_tasks = list()
        self.tasks = PriorityQ()
        self.active_tasks = PriorityQ()
        self.completed_tasks = list()

        self.resource_pool = ResourcePool()

        if topology is not None:
            self.set_topology(topology)

    def set_graphs(self, compute_graph, movement_graph):
        self.compute_graph = compute_graph
        self.movement_graph = movement_graph

    def set_data_graph(self, data_graph):
        self.data_graph = data_graph

    def set_task_dictionaries(self, task_dictionaries, movement_dictionary):
        self.task_dictionaries = task_dictionaries
        self.movement_dictionary = movement_dictionary

    def set_order(self, order):
        self.order = order

    def set_mapping(self, mapping):
        self.mapping = mapping

    def set_state(self, state):
        self.state = state

    def __str__(self):
        return f"Schedule {self.name} | Current Time: {self.time} | \n\t Unscheduled Tasks: {self.tasks.queue} | \n\t Active Tasks: {self.active_tasks.queue} | \n\t Completed Tasks: {self.completed_tasks} | \n\t Resource Pool: {self.resource_pool.pool} | \n\t Devices: {[v for i, v in self.devicespace.items()]}"

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)

    def set_topology(self, topology):
        self.resource_pool.set_toplogy(topology)
        self.devices = topology.devices
        for device in topology.devices:
            self.devicespace[device.name] = device

    def set_data(self, data_list):
        self.data = data_list
        # for data in data_list:
        #    self.dataspace[data.name] = data

    def set_tasks(self, task_list):
        self.all_tasks = task_list

        if type(self.tasks) == type(task_list):
            self.tasks = task_list
        else:
            for task in task_list:
                self.tasks.put(task)
        # for task in task_list:
        #    self.taskspace[task.name] = task

    def adjust_references(self, taskspace):
        for task in self.all_tasks:
            task.convert_references(taskspace)

        # for data in self.data:
        #    data.dataspace = self.dataspace

    def add_device(self, device):
        # TODO: Deprecated (use set_topology)
        self.resource_pool.add_resources(device.name, device.resources)
        device.active_memory = self.resource_pool
        self.devicespace[device.name] = device

    def remove_device(self, device):
        # TODO: Deprecated (use set_topology)
        self.resource_pool.delete_device(device.name)
        del self.devicespace[device.name]

    def assign_all_tasks(self):
        idx = 0
        for task in Drainer(self.tasks):
            for location in task.locations:
                device = self.devicespace[location]
                task.order = idx
                idx = idx+2
                device.add_planned_task(task)
                self.state.set_task_status(task.name, "status", "mapped")
                self.state.set_task_status(task.name, "device", device.id)
            # self.all_tasks.append(task)

    def complete_task(self):
        recent_task = self.active_tasks.get()[1]
        if recent_task:

            print("Finished Task", recent_task.name)

            # Stop reserving memory
            recent_task.free_resources(self.resource_pool)

            # Remove from active device queues
            for device in recent_task.locations:
                self.devicespace[device].pop_local_active_task()

            # Update dependencies
            recent_task.finish()

            # Unlock used data (so it can possibly be evicted)
            # recent_task.unlock_data()

            # Advance global time
            self.time = max(recent_task.completion_time, self.time)

            # Add to completed tasks
            self.completed_tasks.append(recent_task)

            # Update global state log
            self.state.advance_time(self.time)
            self.state.set_task_status(
                recent_task.name, "completion_time", self.time)
            self.state.set_task_status(
                recent_task.name, "status", "completed")

    def start_tasks(self):
        for device in self.devices:
            for task in Assign2(device.planned_compute_tasks, device.planned_movement_tasks, self.resource_pool):

                if task:
                    print("Assigning Task:", task.name, task.dependencies)

                    # data0 = task.read_data[0]
                    # print(data0)

                    # Assign memory
                    task.reserve_resources(self.resource_pool)

                    # Compute time to completion
                    task.estimate_time(self.time, self.resource_pool)
                    # print("Expected Complete: ", task.completion_time)
                    # "Start" Task
                    # 1. Locks data (so it can't be evicted)
                    # 2. Updates data status (to evict stale data)
                    task.start()

                    # Lock used data
                    # task.lock_data()

                    # Push task to global active queue
                    self.active_tasks.put((task.completion_time, task))

                    # Push task to local active queue (for device)
                    # NOTE: This is mainly just as a convience for logging
                    device.add_local_active_task(task)

                    # data0 = task.read_data[0]
                    # print(data0)

                    # Update global state log
                    self.state.set_task_status(
                        task.name, "status", "active")
                    self.state.set_task_status(
                        task.name, "start_time", self.time)
                else:
                    continue

        for device in self.devices:
            self.state.log_device(device)

        for data in self.data:
            self.state.log_data(data)

    def count_planned_tasks(self, type="Both"):
        count_compute = 0
        count_move = 0
        for device in self.devicespace.values():
            count_compute += len(device.planned_compute_tasks)
            count_move += len(device.planned_movement_tasks)

        if type == "Compute":
            return count_compute
        elif type == "Move":
            return count_move
        elif type == "Both":
            return count_compute + count_move
        else:
            raise Exception(
                "Valid counting types are 'Compute', 'Move', and 'Both'")

    def count_remaining_tasks(self):
        return len(self.all_tasks) - len(self.completed_tasks)

    def get_task_by_name(self, name):
        for task in self.all_tasks:
            if task.name == name:
                return task
        return None

    def get_completed_task_by_name(self, name):
        for task in self.completed_tasks:
            if task.name == name:
                return task
        return None

    def run(self):
        # Map all tasks to devices
        self.assign_all_tasks()

        i = 0
        # gpu0 = self.devicespace['gpu0']
        # print("initial:")
        # print(gpu0)

        # Start tasks
        self.start_tasks()

        # print(i, "| Push")
        # print(gpu0)

        print("Time: ", self.time, "Remaining: ", self.count_remaining_tasks(
        ), "Running: ", len(self.active_tasks.queue))
        # print(gpu0.active_data.pool[0])

        # Run until all tasks are complete
        while self.count_remaining_tasks() > 0:

            self.complete_task()
            # print(i, "| Pull")
            # print(gpu0)
            self.start_tasks()

            print("Time: ", self.time, "Remaining: ", self.count_remaining_tasks(),
                  "Running: ", len(self.active_tasks.queue))

            # print(i, "| Push")
            # print(gpu0.active_data.pool[0])

            for device in self.devices:
                print(device)

            print("-----------")

        return self.time


def initialize_task_handles(graph, task_dictionaries, movement_dictionaries, device_map, data_map):
    runtime_dict, dependency_dict, write_dict, read_dict, count_dict = task_dictionaries
    data_tasks, data_task_dict, task_to_movement_dict = movement_dictionaries

    task_handle_dict = dict()
    # __init__(self, task_id, runtime, dependency,
    #         dependants, read, write, associated_movement)

    TaskHandle.set_data(data_map)
    TaskHandle.set_devices(device_map)
    TaskHandle.set_movement_dict(data_task_dict)

    for task_name in graph.nodes():
        name = task_name

        if "M" in name:
            continue

        runtime = runtime_dict[name]
        dependency = dependency_dict[name]

        in_edges = graph_full.in_edges(nbunch=[name])
        in_edges = [edge[0] for edge in in_edges]

        out_edges = graph_full.out_edges(nbunch=[name])
        out_edges = [edge[1] for edge in out_edges]

        reads = read_dict[name]
        writes = write_dict[name]

        movement = task_to_movement_dict[name]

        task = TaskHandle(task_name, runtime, in_edges,
                          out_edges, reads, writes, movement)
        task_handle_dict[task_name] = task

    return task_handle_dict


def instantiate_tasks(task_handles, task_list, mapping):
    task_dict = dict()

    for task_name in task_list:

        task_handle = task_handles[task_name]
        device = mapping[task_name]

        current_task = task_handle.make_task(device)
        task_dict[task_name] = current_task

        movement_tasks = task_handle.make_movement_tasks(device)

        for task in movement_tasks:
            task_dict[task.name] = task

    return task_dict


def order_tasks(tasks, order):
    task_list = []
    idx = 0
    for task_name in order:
        task_list.append(tasks[task_name])
        task_list[idx].order = idx
        idx = idx + 1
    return task_list


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
                memory = state[device.name].available_memory(usage=True)
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
                memory = state[device.name].available_acus(usage=True)
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
        point = state.get_state_at_time(time_stamp)
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
        plot_graph(graph_full, state.get_state_at_time(
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


data_config, task_list = read_graphx("test.gphx")  #
#data_config, task_list = read_graphx("reduce.gph")
runtime_dict, dependency_dict, write_dict, read_dict, count_dict = convert_to_dictionary(
    task_list)

print(dependency_dict)

task_dictionaries = (runtime_dict, dependency_dict,
                     write_dict, read_dict, count_dict)

task_data_dependencies = compute_data_edges(
    (runtime_dict, dependency_dict, write_dict, read_dict, count_dict))

data_tasks, data_task_dict, task_to_movement_dict = add_data_tasks(
    task_list, (runtime_dict, dependency_dict, write_dict, read_dict, count_dict), task_data_dependencies)

movement_dictionaries = (data_tasks, data_task_dict, task_to_movement_dict)

graph_compute = make_networkx_graph(task_list, (runtime_dict, dependency_dict, write_dict, read_dict,
                                                count_dict), data_config, None)
graph_w_data = make_networkx_graph(task_list, (runtime_dict, dependency_dict, write_dict, read_dict,
                                               count_dict), data_config, (data_tasks, data_task_dict, task_to_movement_dict))

graph_full = make_networkx_graph(task_list, (runtime_dict, dependency_dict, write_dict, read_dict,
                                             count_dict), data_config, (data_tasks, data_task_dict, task_to_movement_dict), check_redundant=False)

#hyper, hyper_dual = make_networkx_datagraph(task_list, (runtime_dict, dependency_dict, write_dict, read_dict,
#                                                        count_dict), data_config, (data_tasks, data_task_dict, task_to_movement_dict))


plot_graph(graph_full, data_dict=(read_dict, write_dict, dependency_dict))
#plot_hypergraph(hyper_dual)

# Create devices
gpu0 = SyntheticDevice(
    "gpu0", 0, {"memory": parse_size("16 GB"), "acus": Fraction(1)})
gpu1 = SyntheticDevice(
    "gpu1", 1, {"memory": parse_size("16 GB"), "acus": Fraction(1)})
gpu2 = SyntheticDevice(
    "gpu2", 2, {"memory": parse_size("16 GB"), "acus": Fraction(1)})
gpu3 = SyntheticDevice(
    "gpu3", 3, {"memory": parse_size("16 GB"), "acus": Fraction(1)})
cpu = SyntheticDevice(
    "cpu", -1, {"memory": parse_size("40 GB"), "acus": Fraction(1)})

# Create device topology
topology = SyntheticTopology("Top1", [gpu0, gpu1, gpu2, gpu3, cpu])

bw = 100
topology.add_connection(gpu0, gpu1, symmetric=True)
topology.add_connection(gpu2, gpu3, symmetric=True)

topology.add_bandwidth(gpu0, gpu1, 2*bw, reverse=bw)
topology.add_bandwidth(gpu0, gpu2, bw, reverse=bw)
topology.add_bandwidth(gpu0, gpu3, bw, reverse=bw)

topology.add_bandwidth(gpu1, gpu2, bw, reverse=bw)
topology.add_bandwidth(gpu1, gpu3, bw, reverse=bw)

topology.add_bandwidth(gpu2, gpu3, 2*bw, reverse=bw)

# Self copy (not used)
topology.add_bandwidth(gpu3, gpu3, bw, reverse=bw)
topology.add_bandwidth(gpu2, gpu2, bw, reverse=bw)
topology.add_bandwidth(gpu1, gpu1, bw, reverse=bw)
topology.add_bandwidth(gpu0, gpu0, bw, reverse=bw)
topology.add_bandwidth(cpu, cpu, bw, reverse=bw)

# With CPU
topology.add_bandwidth(gpu0, cpu, bw, reverse=bw)
topology.add_bandwidth(gpu1, cpu, bw, reverse=bw)
topology.add_bandwidth(gpu2, cpu, bw, reverse=bw)
topology.add_bandwidth(gpu3, cpu, bw, reverse=bw)

# Initialize Scheduler
scheduler = SyntheticSchedule("Scheduler", topology=topology)

devices = scheduler.devices
device_map = form_device_map(devices)
data_map = initialize_data(data_config, device_map)
data = list(data_map.values())

#Level = 0 (Save minimal state)
#Level = 1 (Save everything! Deep Copy of objects! Needed for some plotting functions below.)
state = State(graph_full, level=1)

task_handles = initialize_task_handles(graph_full, task_dictionaries,
                                       movement_dictionaries, device_map, data_map)

order = get_valid_order(graph_compute)
mapping = get_trivial_mapping(task_handles)

tasks = instantiate_tasks(task_handles, order, mapping)
full_order = get_valid_order(graph_full)
task_list = order_tasks(tasks, full_order)

scheduler.set_tasks(task_list)
scheduler.adjust_references(tasks)
scheduler.set_state(state)
scheduler.set_data(list(data_map.values()))

for d in data:
    print(d)

t = time.perf_counter()
graph_t = scheduler.run()
t = time.perf_counter() - t
print("Simulator Took (s): ", t)
print("Predicted Graph Time (s): ", graph_t)


#import sys
#sys.exit(0)

#point = state.get_state_at_time(0.5)
#print(point)
#plot_graph(graph_full, state.active_state, task_device_color_map)

# make_image_folder("reduce_graphs", state)
# make_image_folder_time(scheduler.time, 0.04166, "reduce_graphs_time", state)

#plot_memory(devices, state)
#plot_active_tasks(devices, state, "all_useful")
#plot_transfers_data(data[0], devices, state)
# make_interactive(scheduler.time, state)

