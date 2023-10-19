from ..types import Architecture, Device, TaskID, DataID, DataInfo, ResourceType, Time
from typing import List, Dict, Set, Tuple, Optional, Callable, Sequence, Type
from .device import SimulatedDevice, ResourceSet
from dataclasses import dataclass, field, InitVar
from .utility import parse_size

import numpy as np
from fractions import Fraction

NamedDevice = Device | SimulatedDevice


@dataclass(slots=True)
class ConnectionPool:
    host: SimulatedDevice
    devices: InitVar[Sequence[Device]]
    devices2index: Dict[Device, int] = field(init=False)
    connections: np.ndarray = field(init=False)
    active_connections: np.ndarray = field(init=False)
    bandwidth: np.ndarray = field(init=False)

    def __post_init__(self, devices: Sequence[Device]):
        """
        Create a connection pool for a set of devices.
        """
        self.devices2index = {}
        for i, device in enumerate(devices):
            self.devices2index[device] = i

        self.connections = np.zeros((len(devices), len(devices)), dtype=np.bool_)
        self.active_connections = np.zeros(
            (len(devices), len(devices)), dtype=np.uint32
        )
        self.bandwidth = np.zeros((len(devices), len(devices)), dtype=np.float32)

    def get_index(self, device: NamedDevice) -> int:
        if isinstance(device, SimulatedDevice):
            device = device.name
        return self.devices2index[device]

    def get_indicies(self, devices: Sequence[NamedDevice]) -> Sequence[int]:
        return [self.get_index(device) for device in devices]

    def check_connection_exists(self, source: NamedDevice, target: NamedDevice):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        return self.connections[source_idx, target_idx]

    def check_bandwidth_exists(self, source: NamedDevice, target: NamedDevice):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        return self.bandwidth[source_idx, target_idx] > 0

    def get_bandwidth(self, source: NamedDevice, target: NamedDevice):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        return self.bandwidth[source_idx, target_idx]

    def count_active_connections(
        self, source: NamedDevice, target: Optional[NamedDevice] = None
    ):
        source_idx = self.get_index(source)

        if target is not None:
            target_idx = self.get_index(target)
            return self.active_connections[source_idx, target_idx]
        else:
            return np.sum(self.active_connections[source_idx, :])

    def add_bandwidth(
        self,
        source: NamedDevice,
        target: NamedDevice,
        bandwidth: float,
        bidirectional: bool = True,
    ):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)

        self.bandwidth[source_idx, target_idx] = bandwidth

        if bidirectional:
            self.bandwidth[target_idx, source_idx] = bandwidth

    def add_connection(
        self, source: NamedDevice, target: NamedDevice, bidirectional: bool = True
    ):
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)

        self.connections[source_idx, target_idx] = True

        if bidirectional:
            self.connections[target_idx, source_idx] = True

    def update_connection_usage(
        self,
        source: NamedDevice,
        target: NamedDevice,
        value: int,
        bidirectional: bool = False,
    ) -> bool:
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)

        if source_idx == target_idx:
            # No connections needed
            return False

        if isinstance(source, SimulatedDevice):
            source = source.name
        if isinstance(target, SimulatedDevice):
            target = target.name

        self.active_connections[source_idx, target_idx] += value

        if bidirectional:
            self.active_connections[source_idx, target_idx] += value

        if self.connections[source_idx, target_idx] <= 0:
            # If no direct connection, route through the host device
            host_idx = self.get_index(self.host)

            self.active_connections[source_idx, host_idx] += value
            self.active_connections[host_idx, target_idx] += value

            if bidirectional:
                self.active_connections[target_idx, host_idx] += value
                self.active_connections[host_idx, source_idx] += value

        return True

    def acquire_connection(self, src: NamedDevice, dst: NamedDevice):
        self.update_connection_usage(src, dst, 1)

    def release_connection(self, src: NamedDevice, dst: NamedDevice):
        self.update_connection_usage(src, dst, -1)

    def check_connection_available(
        self,
        source: SimulatedDevice,
        target: SimulatedDevice,
        require_copy_engines: bool = True,
        require_symmetric=True,
    ) -> bool:
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)

        if source_idx == target_idx:
            # No connection needed for a self copy
            return True

        # Is there a direct connection?
        direct_connection = self.check_connection_exists(source, target)

        # Check if copy engines are available (if required)
        if require_copy_engines:
            if (
                self.count_active_connections(source)
                >= source.resources.store[ResourceType.COPY]
            ):
                return False
            if (
                require_symmetric
                and self.count_active_connections(target)
                >= target.resources.store[ResourceType.COPY]
            ):
                return False

            if not direct_connection:
                if not self.check_connection_exists(source, self.host):
                    return False
                if not self.check_connection_exists(target, self.host):
                    return False

                if (
                    self.count_active_connections(self.host)
                    >= self.host[ResourceType.COPY]
                ):
                    return False
        return False

    def sort_by_bandwidth(
        self, target: NamedDevice, devices: Sequence[NamedDevice]
    ) -> Sequence[NamedDevice]:
        """
        Return a sorted list of devices by the bandwidth of the connection to the target
        """
        target_idx = self.get_index(target)
        bandwidths = self.bandwidth[target_idx, self.get_indicies(devices)]
        return [devices[i] for i in np.argsort(bandwidths)]

    def get_transfer_time(
        self, source: NamedDevice, target: NamedDevice, data_size: int
    ) -> Time:
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        bandwidth = self.bandwidth[source_idx, target_idx]
        time_in_seconds = data_size / bandwidth
        time_in_microseconds = int(time_in_seconds * 1e6)
        return Time(time_in_microseconds)

    def get_connection_string(self, source: NamedDevice, target: NamedDevice) -> str:
        source_idx = self.get_index(source)
        target_idx = self.get_index(target)
        return f"{source} -> {target} (bw={self.bandwidth[source_idx, target_idx]}, active={self.active_connections[source_idx, target_idx]})"

    def __str__(self) -> str:
        s = "ConnectionPool:\n"
        for source in self.devices2index.keys():
            for target in self.devices2index.keys():
                if self.check_connection_exists(source, target):
                    s += self.get_connection_string(source, target) + "\n"
        return s

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(slots=True)
class SimulatedTopology:
    devices: List[SimulatedDevice]
    name: str = "SimulatedTopology"
    connection_pool: ConnectionPool = field(init=False)

    def __post_init__(self):
        """
        Create a simulated topology.
        Assumes that the first device is the host device.
        """
        device_names = [device.name for device in self.devices]
        self.connection_pool = ConnectionPool(
            host=self.devices[0], devices=device_names
        )

    def add_connection(
        self, source: NamedDevice, target: NamedDevice, bidirectional: bool = True
    ):
        self.connection_pool.add_connection(source, target, bidirectional)

    def add_bandwidth(
        self,
        source: NamedDevice,
        target: NamedDevice,
        bandwidth: float,
        bidirectional: bool = True,
    ):
        self.connection_pool.add_bandwidth(source, target, bandwidth, bidirectional)

    def check_connection_exists(self, source: NamedDevice, target: NamedDevice):
        return self.connection_pool.check_connection_exists(source, target)

    def check_bandwidth_exists(self, source: NamedDevice, target: NamedDevice):
        return self.connection_pool.check_bandwidth_exists(source, target)

    def get_bandwidth(self, source: NamedDevice, target: NamedDevice):
        return self.connection_pool.get_bandwidth(source, target)

    def count_active_connections(
        self, source: NamedDevice, target: Optional[NamedDevice] = None
    ):
        return self.connection_pool.count_active_connections(source, target)

    def acquire_connection(self, src: NamedDevice, dst: NamedDevice):
        self.connection_pool.acquire_connection(src, dst)

    def release_connection(self, src: NamedDevice, dst: NamedDevice):
        self.connection_pool.release_connection(src, dst)

    def check_connection_available(
        self,
        source: SimulatedDevice,
        target: SimulatedDevice,
        require_copy_engines: bool = True,
        require_symmetric=True,
    ) -> bool:
        return self.connection_pool.check_connection_available(
            source, target, require_copy_engines, require_symmetric
        )

    def nearest_valid_connection(
        self,
        target: SimulatedDevice,
        sources: Sequence[SimulatedDevice],
        require_copy_engines: bool = True,
        require_symmetric=True,
    ) -> Optional[NamedDevice]:
        sorted_sources = self.connection_pool.sort_by_bandwidth(target, sources)
        for source in sorted_sources:
            assert isinstance(source, SimulatedDevice)

            if self.check_connection_available(
                source, target, require_copy_engines, require_symmetric
            ):
                return source
        return None

    def get_devices(self, device_type: Architecture) -> List[SimulatedDevice]:
        return [
            device for device in self.devices if device.name.architecture == device_type
        ]

    def get_device_string(self, device: SimulatedDevice) -> str:
        return f"{device} (mem={device[ResourceType.MEMORY]})"

    def __str__(self) -> str:
        s = f"Topology: {self.name}\n"
        for device in self.devices:
            s += self.get_device_string(device) + "\n"
        s += str(self.connection_pool)
        return s

    def __repr__(self) -> str:
        return self.__str__()


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
    topology = SimulatedTopology(cpus + gpus, "Topology::4G-1C")

    for gpu in gpus:
        topology.add_connection(gpu, cpus[0], bidirectional=True)
        topology.add_bandwidth(gpu, cpus[0], D2H_BW)
        topology.add_bandwidth(cpus[0], gpu, H2D_BW)

    topology.add_connection(gpus[0], gpus[1], bidirectional=True)
    topology.add_bandwidth(gpus[0], gpus[1], P2P_BW)

    topology.add_connection(gpus[2], gpus[3], bidirectional=True)
    topology.add_bandwidth(gpus[2], gpus[3], P2P_BW)

    return topology
