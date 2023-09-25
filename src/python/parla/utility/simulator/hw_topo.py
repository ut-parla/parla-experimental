import numpy as np
import cupy as cp

from data import PArray

class BandwidthHandle(object):

    def create_parray(source, size):
        """
        Create PArrays using Cupy arrays to get average data transfer time
        between devices.
        """
        if source.idx >=0:
            with cupy.cuda.Device(source.idx) as device:
                data = cp.ones(size, dtype=cp.float32)
                device.synchronize()
        else:
            data = np.ones(size, dtype=np.float32)
        return data

    def copy(self, arr, source, destination):
        """
        Copy parray transfer time between devices.
        """
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
        """
        Estimate actual Cupy array transfer times.
        """
        times= []
        for i in range(samples):
            parray = self.create_parray(source, size)
            start = time.perf_counter()
            self.copy(parray, source, destination)
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

        # Device X Device connections and bandwidths
        if bandwidth is None:
            bandwidth = np.zeros((len(self.devices), len(self.devices)))
        self.bandwidth = bandwidth

        if connections is None:
            connections = np.zeros(
                (len(self.devices), len(self.devices)), dtype=np.int32)

        self.connections = connections

        # Number of connections between devices
        self.active_connections = np.zeros(
            (len(self.devices), len(self.devices)), dtype=np.int32)

        # Set backlink to devices (just in case we ever need to query it)
        for device in self.devices:
            device.topology = self
            # All devices are connected to the host
            self.add_connection(self.host, device, symmetric=True)

            # All devices are connected to themselves
            self.add_connection(device, device)

        # Number of copy engines
        self.max_num_copy_engines = dict()
        self.num_active_copy_engines = dict()
        for d in device_list:
            self.max_num_copy_engines[d.name] = d.num_copy_engines
            self.num_active_copy_engines[d.name] = 0

    def add_bandwidth(self, device1, device2, value, reverse_value=None):
        """
        Add bandwidth between device1 and device2.
        """
        self.bandwidth[device1.idx, device2.idx] = value

        if reverse_value is not None:
            self.bandwidth[device2.idx, device1.idx] = reverse_value

    def get_device_by_id(self, idx):
        """
        Get device object by device id.
        """
        return self.id_map[idx]

    def sample_bandwidth(self, device1, device2, size=10**6, samples=20):
        """
        Estimate bandwidth between device1 and device2.
        """
        self.bandwidth[device1.idx, device2.idx] = BandwidthHandle.estimate(
            device1, device2, size, samples)
        self.bandwidth[device2.idx, device1.idx] = BandwidthHandle.estimate(
            device2, device1, size, samples)

    def fill_bandwidth(self, size=10**6, samples=20):
        """
        Fill bandwidth information.
        """
        for device1 in self.devices:
            for device2 in self.devices:
                self.sample_bandwidth(device1, device2, size, samples)

    def add_connection(self, device1, device2, symmetric=True):
        """
        Add connection between devices.
        """
        if symmetric:
            self.connections[device1.idx, device2.idx] = 1
            self.connections[device2.idx, device1.idx] = 1
        else:
            self.connections[device1.idx, device2.idx] = 1

    def remove_connection(self, device1, device2, symmetric=True):
        """
        Remove connection between devices.
        """
        if symmetric:
            self.connections[device1.idx, device2.idx] = 0
            self.connections[device2.idx, device1.idx] = 0
        else:
            self.connections[device1.idx, device2.idx] = 0

    def acquire_device_links(self, device1, device2):
        """
        Use device links between device1 and device2; possibly
        passing through CPU like through QPI.
        This method updates the related counters correspondingly.
        """
        self.active_connections[device1.idx, device2.idx] += 1
        # self.active_connections[device2][device1] += 1

        if device1 == device2:
            return

        self.num_active_copy_engines[device1.name] += 1
        self.num_active_copy_engines[device2.name] += 1

        if self.connections[device1.idx, device2.idx] <= 0:
            # Assume communication is through CPU buffer
            self.active_connections[device1.idx, self.host.idx] += 1
            self.active_connections[self.host.idx, device2.idx] += 1
            self.num_active_copy_engines[self.host.name] += 1

    def release_device_links(self, device1, device2):
        """
        Release device link between device1 and device2.
        """
        self.active_connections[device1.idx, device2.idx] -= 1
        # self.active_connections[device2][device1] -= 1

        if device1 == device2:
            return

        self.num_active_copy_engines[device1.name] -= 1
        self.num_active_copy_engines[device2.name] -= 1

        if self.connections[device1.idx][device2.idx] <= 0:
            # Assume communication is through CPU buffer
            self.active_connections[device1.idx, self.host.idx] -= 1
            self.active_connections[self.host.idx, device2.idx] -= 1
            self.num_active_copy_engines[self.host.name] -= 1

    def check_link_availability(
        self, device1, device2, symmetry=True, engines=True):
        """
        Check if two devices have available links and copy engines.
        """

        if device1 == device2:
            return True

        if engines:
            if self.num_active_copy_engines[device1.name] >= \
               self.max_num_copy_engines[device1.name]:
                return False
            if self.num_active_copy_engines[device2.name] >= \
               self.max_num_copy_engines[device2.name]:
                return False

        # TODO: Will need to be adjusted for connections that can support more than 1 simultaneous copy

        if symmetry:
            return self.active_connections[device1.idx, device2.idx] == 0 and \
                   self.active_connections[device2.idx, device1.idx] == 0
        else:
            return self.active_connections[device1.idx, device2.idx] == 0

    def find_best_parray_source(self, target, source_list, free=True):
        """
        Find best device source for parray to move by considering
        link status.
        """
        if free:
            # Find the closest free source for the data
            closest_free_source = None
            closest_free_distance = np.inf
            for source in source_list:
                if self.check_link_availability(
                    target, source, symmetry=True, engines=True):
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
        """
        Compute PArray transfer time between device source and target.
        """
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

    def reserve_communication(self, parray_targets, parray_sources):
        """
        Reserve the communication links for PArray movement.
        """
        reserved = dict()

        for parray_name in parray_targets.keys():
            target_device = parray_targets[parray_name]
            source_device = parray_sources[parray_name]
            data = PArray.dataspace[parray_name]

            # Only take resources if the transfer is not already active
            if (not data.active_transfer(target_device)) and (not (target_device == source_device)):
                reserved[parray_name] = True
                self.acquire_device_links(target_device, source_device)
                data.add_transfer(target_device)

            return reserved

    def free_communication(self, parray_targets, parray_sources, reserved):
        """
        Free all communication links between devices used for PArray transfers.
        """
        for parray_name in parray_targets.keys():
            target_device = parray_targets[parray_name]
            source_device = parray_sources[parray_name]
            data = PArray.dataspace[parray_name]

            # Only free resources if the transfer is not already active
            if parray_name in reserved:
                self.release_device_links(target_device, source_device)

    def __str__(self):
        return f"Topology: {self.name} | \n\t Devices: {self.devices} | \n\t Bandwidth: {self.bandwidth} | \n\t Connections: {self.connections}"

    def __repr__(self):
        return str(self.name)

    def __hash__(self):
        return hash(self.name)

