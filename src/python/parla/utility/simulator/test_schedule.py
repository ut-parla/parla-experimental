import numpy as np
import time
import warnings
import queue
from fractions import Fraction

import device
from resource_pool import ResourcePool
from priority_queue import PriorityQueue
from data import PArray
from utility import read_graphx, convert_to_dictionary, compute_data_edges
from utility import add_data_tasks, make_networkx_graph, get_valid_order
from utility import task_status_color_map, plot_graph
from task import TaskHandle, SyntheticTask
from log_stack import LogDict
from hw_topo import BandwidthHandle, SyntheticTopology

SyntheticDevice = device.SyntheticDevice

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
        data = PArray("D"+str(data_name),
                      data_info[0], [device])
        device.add_data(data)
        data_list[data.name] = data
    return data_list



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
        self.tasks = PriorityQueue()
        self.active_tasks = PriorityQueue()
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
            task.convert_tasknames_to_objs(taskspace)

        # for data in self.data:
        #    data.dataspace = self.dataspace

    def add_device(self, device):
        # TODO: Deprecated (use set_topology)
        self.resource_pool.add_resources(device.name, device.resources)
        device.resource_pool = self.resource_pool
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
                device.push_planned_task(task)
                self.state.set_task_log(task.name, "status", "mapped")
                self.state.set_task_log(task.name, "device", device.id)
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
            self.state.set_task_log(
                recent_task.name, "completion_time", self.time)
            self.state.set_task_log(
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
                    device.push_local_active_task(task)

                    # data0 = task.read_data[0]
                    # print(data0)

                    # Update global state log
                    self.state.set_task_log(
                        task.name, "status", "active")
                    self.state.set_task_log(
                        task.name, "start_time", self.time)
                else:
                    continue

        for device in self.devices:
            self.state.log_device(device)

        for data in self.data:
            self.state.log_parray(data)

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
    data_tasks, datamove_task_meta_info, compute_tid_to_datamove_tid = movement_dictionaries

    task_handle_dict = dict()
    # __init__(self, task_id, runtime, dependency,
    #         dependants, read, write, associated_movement)

    TaskHandle.set_parray_name_to_obj(data_map)
    TaskHandle.set_device_id_to_obj(device_map)
    TaskHandle.set_datamove_task_meta_info(datamove_task_meta_info)

    for task_name in graph.nodes():
        if "M" in task_name:
            continue

        runtime = runtime_dict[task_name]
        dependency = dependency_dict[task_name]

        in_edges = graph_full.in_edges(nbunch=[task_name])
        in_edges = [edge[0] for edge in in_edges]

        out_edges = graph_full.out_edges(nbunch=[task_name])
        out_edges = [edge[1] for edge in out_edges]

        reads = read_dict[task_name]
        writes = write_dict[task_name]

        # Extract a list of ids of data move tasks for a task
        # having `name.`
        datamove_tid_list = compute_tid_to_datamove_tid[
            task_name]

        task = TaskHandle(task_name, runtime, in_edges,
                          out_edges, reads, writes,
                          datamove_tid_list)
        task_handle_dict[task_name] = task

    return task_handle_dict


def instantiate_tasks(task_handles, task_list, mapping):
    task_dict = dict()

    for task_name in task_list:

        task_handle = task_handles[task_name]
        device = mapping[task_name]

        current_task = task_handle.create_compute_task(device)
        task_dict[task_name] = current_task

        movement_tasks = task_handle.create_movement_tasks(device)

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


data_config, task_list = read_graphx("test.gphx")  #
#data_config, task_list = read_graphx("reduce.gph")
runtime_dict, dependency_dict, write_dict, read_dict, count_dict = convert_to_dictionary(
    task_list)

print(dependency_dict)

task_dictionaries = (runtime_dict, dependency_dict,
                     write_dict, read_dict, count_dict)

task_data_dependencies = compute_data_edges(
    (runtime_dict, dependency_dict, write_dict, read_dict, count_dict))

data_tasks, datamove_task_meta_info, compute_tid_to_datamove_tid = add_data_tasks(
    task_list, (runtime_dict, dependency_dict, write_dict, read_dict, count_dict), task_data_dependencies)

movement_dictionaries = (data_tasks, datamove_task_meta_info, compute_tid_to_datamove_tid)

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

topology.add_bandwidth(gpu0, gpu1, 2*bw, reverse_value=bw)
topology.add_bandwidth(gpu0, gpu2, bw, reverse_value=bw)
topology.add_bandwidth(gpu0, gpu3, bw, reverse_value=bw)

topology.add_bandwidth(gpu1, gpu2, bw, reverse_value=bw)
topology.add_bandwidth(gpu1, gpu3, bw, reverse_value=bw)

topology.add_bandwidth(gpu2, gpu3, 2*bw, reverse_value=bw)

# Self copy (not used)
topology.add_bandwidth(gpu3, gpu3, bw, reverse_value=bw)
topology.add_bandwidth(gpu2, gpu2, bw, reverse_value=bw)
topology.add_bandwidth(gpu1, gpu1, bw, reverse_value=bw)
topology.add_bandwidth(gpu0, gpu0, bw, reverse_value=bw)
topology.add_bandwidth(cpu, cpu, bw, reverse_value=bw)

# With CPU
topology.add_bandwidth(gpu0, cpu, bw, reverse_value=bw)
topology.add_bandwidth(gpu1, cpu, bw, reverse_value=bw)
topology.add_bandwidth(gpu2, cpu, bw, reverse_value=bw)
topology.add_bandwidth(gpu3, cpu, bw, reverse_value=bw)

# Initialize Scheduler
scheduler = SyntheticSchedule("Scheduler", topology=topology)

devices = scheduler.devices
device_map = form_device_map(devices)
data_map = initialize_data(data_config, device_map)
data = list(data_map.values())

graph_compute = make_networkx_graph(task_list, (runtime_dict, dependency_dict, write_dict, read_dict,
                                                count_dict), data_config, None)
graph_w_data = make_networkx_graph(task_list, (runtime_dict, dependency_dict, write_dict, read_dict,
                                               count_dict), data_config, (data_tasks, datamove_task_meta_info, compute_tid_to_datamove_tid))
graph_full = make_networkx_graph(task_list, (runtime_dict, dependency_dict, write_dict, read_dict,
                                             count_dict), data_config, (data_tasks, datamove_task_meta_info, compute_tid_to_datamove_tid), check_redundant=False)
#hyper, hyper_dual = make_networkx_datagraph(task_list, (runtime_dict, dependency_dict, write_dict, read_dict,
#                                                        count_dict), data_config, (data_tasks, datamove_task_meta_info, compute_tid_to_datamove_tid))
plot_graph(graph_full, data_dict=(read_dict, write_dict, dependency_dict))
#plot_hypergraph(hyper_dual)

#Level = 0 (Save minimal state)
#Level = 1 (Save everything! Deep Copy of objects! Needed for some plotting functions below.)
state = LogDict(graph_full, log_level=1)

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

#point = state.get_logs_with_time(0.5)
#print(point)
#plot_graph(graph_full, state.active_state, task_device_color_map)

# make_image_folder("reduce_graphs", state)
# make_image_folder_time(scheduler.time, 0.04166, "reduce_graphs_time", state)

#plot_memory(devices, state)
#plot_active_tasks(devices, state, "all_useful")
#plot_transfers_data(data[0], devices, state)
# make_interactive(scheduler.time, state)
