import numpy as np
import time
import warnings
import copy
import queue
from fractions import Fraction
from collections import namedtuple

import device
from resource_pool import ResourcePool
from priority_queue import PriorityQueue
from data import PArray
from utility import read_graphx, convert_to_dictionary, compute_data_edges
from utility import add_data_tasks, make_networkx_graph, get_valid_order
from utility import task_status_color_map, plot_graph

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
            self.max_copy[d.name] = d.num_copy_engines
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
            data = PArray.dataspace[data_name]

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
            data = PArray.dataspace[data_name]

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
            data = PArray.dataspace[data_name]

            # Only free resources if the transfer is not already active
            if data_name in reserved:
                self.decrease_usage(target_device, source_device)


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
            data = PArray.dataspace[data_name]

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

            data.acquire(to_lock)


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

            data.release(to_unlock)

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
            task.convert_references(taskspace)

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
                    device.push_local_active_task(task)

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

