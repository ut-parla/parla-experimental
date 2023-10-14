from ..types import Architecture, Device, TaskID, DataID, DataInfo, ResourceType
from typing import List, Dict, Set, Tuple, Optional, Callable
from .device import SimulatedDevice, ResourceSet
from dataclasses import dataclass
from .utility import parse_size

import numpy as np
from fractions import Fraction

NamedDevice = Device | SimulatedDevice


@dataclass(slots=True)
class SimulatedTopology:
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
        # greater than 0 if used.
        self.connections = np.zeros(
            (len(self.devices), len(self.devices)), dtype=np.int32
        )
        # greater than 0 if used.
        self.active_connections = np.zeros(
            (len(self.devices), len(self.devices)), dtype=np.int32
        )
        # Bandwidth between devices.
        self.bandwidth = np.zeros(
            (len(self.devices), len(self.devices)), dtype=np.float32
        )
        self.active_copy_engines = {device.name: 0 for device in self.devices}
        self.max_copy_engines = {
            device.name: device.resources.store[ResourceType.COPY]
            for device in self.devices
        }

    def __str__(self) -> str:
        repr_str = "[[HW Topology]]\n"
        repr_str += self.name + "\n"
        for d in self.devices:
            repr_str += str(d.name) + "\n"
            repr_str += "[Resource]\n"
            repr_str += "Memory: "
            repr_str += str(d.resources[ResourceType.MEMORY]) + "\n"
            repr_str += "VCU: "
            repr_str += str(d.resources[ResourceType.VCU]) + "\n"
            repr_str += "COPY: "
            repr_str += str(d.resources[ResourceType.COPY]) + "\n\n"

        repr_str += "[Connections]\n"
        for d1 in range(len(self.devices)):
            for d2 in range(len(self.devices)):
                if self.connections[d1, d2] == 1:
                    repr_str += str(self.devices[d1].name) + ","
                    repr_str += str(self.devices[d2].name)
                    repr_str += " bandwidth: " + str(self.bandwidth[d1][d2])
                    repr_str += "\n"

        return repr_str

    def __repr__(self) -> str:
        return self.__str__()

    def get_index(self, device: NamedDevice) -> int:
        if isinstance(device, SimulatedDevice):
            device = device.name
        return self.id_map[device]

    def add_bandwidth(
        self,
        src: NamedDevice,
        dst: NamedDevice,
        bandwidth: float,
        bidirectional: bool = True,
    ):
        src_idx = self.get_index(src)
        dst_idx = self.get_index(dst)

        self.bandwidth[src_idx, dst_idx] = bandwidth

        if bidirectional:
            self.bandwidth[dst_idx, src_idx] = bandwidth

    def add_connection(
        self, src: NamedDevice, dst: NamedDevice, bidirectional: bool = True
    ):
        src_idx = self.get_index(src)
        dst_idx = self.get_index(dst)

        self.connections[src_idx, dst_idx] = 1

        if bidirectional:
            self.connections[dst_idx, src_idx] = 1

    def update_connections(
        self,
        src: NamedDevice,
        dst: NamedDevice,
        value: int,
        bidirectional: bool = False,
    ):
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

    def check_connection(
        self,
        src: NamedDevice,
        dst: NamedDevice,
        require_engines: bool = True,
        require_symmetric=True,
    ) -> bool:
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
            return (self.active_connections[src_idx, dst_idx] == 0) and (
                self.active_connections[dst_idx, src_idx] == 0
            )
        else:
            return self.active_connections[src_idx, dst_idx] == 0


class TopologyManager:
    generator_map: Dict[str, Callable[[Optional[Dict]], SimulatedTopology]] = {}

    @staticmethod
    def read_from_yaml(topology_name: str) -> SimulatedTopology:
        """
        Read topology from a YAML file.
        """
        raise NotImplementedError

    @staticmethod
    def register_generator(topology_name: str):
        """
        Register a topology generator.
        """

        def decorator(cls):
            if topology_name in TopologyManager.generator_map:
                raise ValueError(
                    f"Topology {topology_name} has already been registered."
                )
            TopologyManager.generator_map[topology_name] = cls
            return cls

        return decorator

    @staticmethod
    def get_generator(
        name: str,
    ) -> Callable[[Optional[Dict]], SimulatedTopology]:
        """
        Get a topology generator.
        """
        if name not in TopologyManager.generator_map:
            raise ValueError(f"Topology {name} is not registered.")
        return TopologyManager.generator_map[name]


@TopologyManager.register_generator("frontera")
def generate_4gpus_1cpu_toplogy(config: Optional[Dict[str, int]]) -> SimulatedTopology:
    """
    This function creates 4 GPUs and 1 CPU architecture.

    The topology looks like below:

    gpu0 - gpu1
     | \   / |
     |  \ /  |
     |  / \  |
     | /   \ |
    gpu2 - gpu3

    gpu0-gpu1 and gpu2-gpu3 have bandwidth of 200 (we assume NVLinks),
    and other connections have bandiwdth of 100.

    All GPUs are connected to CPU by connections having bandwidth of 100.
    Each GPU is equipped with 16GB DRAM, and CPU is equipped with 130GB.
    """

    if config is not None:
        P2P_BW = config["P2P_BW"]
        H2D_BW = config["H2D_BW"]
        D2H_BW = config["D2H_BW"]

        GPU_MEM = config["GPU_MEM"]
        CPU_MEM = config["CPU_MEM"]

        GPU_COPY_ENGINES = config["GPU_COPY_ENGINES"]
        CPU_COPY_ENGINES = config["CPU_COPY_ENGINES"]
    else:
        # Default configuration for testing
        P2P_BW = 200
        H2D_BW = 100
        D2H_BW = 100

        GPU_MEM = parse_size("16 GB")
        CPU_MEM = parse_size("130 GB")

        GPU_COPY_ENGINES = 3
        CPU_COPY_ENGINES = 3

    # Create devices
    gpus = [
        SimulatedDevice(
            Device(Architecture.GPU, i), ResourceSet(1, GPU_MEM, GPU_COPY_ENGINES)
        )
        for i in range(4)
    ]
    cpus = [
        SimulatedDevice(
            Device(Architecture.CPU, 0), ResourceSet(1, CPU_MEM, CPU_COPY_ENGINES)
        )
    ]

    # Create device topology
    topology = SimulatedTopology(gpus + cpus, "Topology::4G-1C")

    for gpu in gpus:
        topology.add_connection(gpu, cpus[0], bidirectional=True)
        topology.add_bandwidth(gpu, cpus[0], D2H_BW)
        topology.add_bandwidth(cpus[0], gpu, H2D_BW)

    topology.add_connection(gpus[0], gpus[1], bidirectional=True)
    topology.add_bandwidth(gpus[0], gpus[1], P2P_BW)

    topology.add_connection(gpus[2], gpus[3], bidirectional=True)
    topology.add_bandwidth(gpus[2], gpus[3], P2P_BW)

    return topology
