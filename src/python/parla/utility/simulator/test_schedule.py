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
from utility import add_data_tasks, make_networkx_graph
from utility import task_status_color_map, plot_graph, Drainer
from utility import initialize_data, form_device_map
import utility
from task import TaskHandle, SyntheticTask
from log_stack import LogDict
from hw_topo import BandwidthHandle, SyntheticTopology
import task

SyntheticDevice = device.SyntheticDevice


class DispatchTaskToDevice(object):
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
            for task in DispatchTaskToDevice(device.planned_compute_tasks,
                                             device.planned_movement_tasks,
                                             self.resource_pool):

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
    "gpu0", 0, {"memory": utility.parse_size("16 GB"), "acus": Fraction(1)})
gpu1 = SyntheticDevice(
    "gpu1", 1, {"memory": utility.parse_size("16 GB"), "acus": Fraction(1)})
gpu2 = SyntheticDevice(
    "gpu2", 2, {"memory": utility.parse_size("16 GB"), "acus": Fraction(1)})
gpu3 = SyntheticDevice(
    "gpu3", 3, {"memory": utility.parse_size("16 GB"), "acus": Fraction(1)})
cpu = SyntheticDevice(
    "cpu", -1, {"memory": utility.parse_size("40 GB"), "acus": Fraction(1)})

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
data_map = utility.initialize_data(data_config, device_map)
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

task_handles = task.initialize_task_handles( \
    graph_full, task_dictionaries, movement_dictionaries, device_map, data_map)

order = utility.get_valid_order(graph_compute)
mapping = utility.get_trivial_mapping(task_handles)

tasks = task.instantiate_tasks(task_handles, order, mapping)
full_order = utility.get_valid_order(graph_full)
task_list = utility.order_tasks(tasks, full_order)

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
