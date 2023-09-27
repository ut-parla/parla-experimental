from ..types import Architecture, Device, TaskID, DataID, DataInfo, ResourceType
from typing import List, Dict, Set, Tuple, Optional
from .device import SimulatedDevice, ResourceSet
from dataclasses import dataclass
from .utility import parse_size

import numpy as np
from fractions import Fraction

NamedDevice = Device | SimulatedDevice


@dataclass(slots=True)
class SimulatedTopology():
    devices: List[SimulatedDevice]
    name: str = "SimulatedTopology"
    id_map: Dict[Device, int] = None
    bandwidth: np.ndarray = None
    host: SimulatedDevice = None

    connections: np.ndarray = None
    active_connections: np.ndarray = None
    active_copy_engines: Dict[Device, int] = None
    max_copy_engines: Dict[Device, int] = None

    def __post_init__(self):
        self.id_map = {}
        for i, device in enumerate(self.devices):
            self.id_map[device.name] = i
            if device.name.architecture == Architecture.CPU:
                self.host = device

        self.active_connections = np.zeros(
            (len(self.devices), len(self.devices)), dtype=np.int32)

        self.active_connections = np.zeros(
            (len(self.devices), len(self.devices)), dtype=np.int32)

        self.bandwidth = np.zeros(
            (len(self.devices), len(self.devices)), dtype=np.float32)

        self.active_copy_engines = {
            device.name: 0 for device in self.devices}
        self.max_copy_engines = {
            device.name: device.resources.store[ResourceType.COPY] \
            for device in self.devices}

    def get_index(self, device: NamedDevice) -> int:
        if isinstance(device, SimulatedDevice):
            device = device.name
        return self.id_map[device]

    def add_bandwidth(self, src: NamedDevice, dst: NamedDevice, bandwidth: float, bidirectional: bool = True):

        src_idx = self.get_index(src)
        dst_idx = self.get_index(dst)

        self.bandwidth[src_idx, dst_idx] = bandwidth

        if bidirectional:
            self.bandwidth[dst_idx, src_idx] = bandwidth

    def add_connection(self, src: NamedDevice, dst: NamedDevice, bidirectional: bool = True):
        src_idx = self.get_index(src)
        dst_idx = self.get_index(dst)

        self.connections[src_idx, dst_idx] = 1

        if bidirectional:
            self.connections[dst_idx, src_idx] = 1

    def update_connections(self, src: NamedDevice, dst: NamedDevice, value: int, bidirectional: bool = False):
        src_idx = self.get_index(src)
        dst_idx = self.get_index(dst)

        if isinstance(src, SimulatedDevice):
            src = src.name
        if isinstance(dst, SimulatedDevice):
            dst = dst.name

        self.active_connections[src_idx, dst_idx] += value

        if bidirectional:
            self.active_connections[dst_idx, src_idx] += value

        if src_idx == dst_idx:
            # No copy engine needed
            return

        self.active_copy_engines[src.name] += value
        self.active_copy_engines[dst.name] += value

        if self.connections[src_idx, dst_idx] <= 0:
            # Assume transfer is through CPU
            # Acquire connections on both ends
            host_idx = self.get_index(self.host)

            self.active_connections[src_idx, host_idx] += value
            self.active_connections[host_idx, dst_idx] += value

            if bidirectional:
                self.active_connections[dst_idx, host_idx] += value
                self.active_connections[host_idx, src_idx] += value

            self.num_active_copy_engines[self.host.name] += value

    def acquire_connection(self, src: NamedDevice, dst: NamedDevice):
        self.update_connections(src, dst, 1)

    def release_connection(self, src: NamedDevice, dst: NamedDevice):
        self.update_connections(src, dst, -1)

    def check_connection(self, src: NamedDevice, dst: NamedDevice, require_engines: bool = True, require_symmetric=True) -> bool:
        src_idx = self.get_index(src)
        dst_idx = self.get_index(dst)

        if isinstance(src, SimulatedDevice):
            src = src.name
        if isinstance(dst, SimulatedDevice):
            dst = dst.name

        if src_idx == dst_idx:
            # No connection needed
            return True

        if require_engines:
            if self.active_copy_engines[src] >= self.max_copy_engines[src]:
                return False
            if self.active_copy_engines[dst] >= self.max_copy_engines[dst]:
                return False

        if require_symmetric:
            return (self.active_connections[src_idx, dst_idx] == 0) \
                and (self.active_connections[dst_idx, src_idx] == 0)
        else:
            return (self.active_connections[src_idx, dst_idx] == 0)



def create_4gpus_1cpu_hwtopo():
    # Create devices
    gpu0 = SimulatedDevice(
        Device(Architecture.GPU, 0), 
        ResourceSet(1, parse_size("16 GB"), 3))
    gpu1 = SimulatedDevice(
        Device(Architecture.GPU, 1), 
        ResourceSet(1, parse_size("16 GB"), 3))
    gpu2 = SimulatedDevice(
        Device(Architecture.GPU, 2), 
        ResourceSet(1, parse_size("16 GB"), 3))
    gpu3 = SimulatedDevice(
        Device(Architecture.GPU, 3), 
        ResourceSet(1, parse_size("16 GB"), 3))
    cpu = SimulatedDevice(
        Device(Architecture.CPU, 4), # TODO(hc): this ID is per-arch? or global?
                                     # what id should be given to here?
        ResourceSet(1, parse_size("16 GB"), 3))

    # Create device topology
    topology = SimulatedTopology([gpu0, gpu1, gpu2, gpu3, cpu], "Top1-4")

    bw = 100
    """
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
    """

    return topology
