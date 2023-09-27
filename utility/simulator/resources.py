from ..types import Architecture, Device, TaskID, DataID, DataInfo, TaskState
from typing import List, Dict, Set, Tuple, Optional
from .device import SimulatedDevice, ResourceSet
from dataclasses import dataclass, InitVar

import numpy as np

NamedDevice = Device | SimulatedDevice


@dataclass(slots=True)
class ResourcePool:
    devices: InitVar[List[SimulatedDevice]]
    pool: Dict[Device, Dict[TaskState, ResourceSet]] = None

    def __post_init__(self, devices: List[SimulatedDevice]):
        self.pool = {}
        for device in devices:
            self.pool[device.name] = {
                TaskState.MAPPED: ResourceSet(*device.resources),
                TaskState.RESERVED: ResourceSet(vcus=0, memory=0, copy=0),
            }

    def add_device_resource(self, device: Device, state: TaskState, resources: ResourceSet):
        self.pool[device][state] += resources

    def remove_device_resources(self, device: Device, state: TaskState, resources: ResourceSet):
        self.pool[device][state] -= resources
        self.pool[device][state].verify()

    def add_resources(self, devices: List[Device], state: TaskState, resources: ResourceSet):
        for device in devices:
            self.add_device_resource(device, state, resources)

    def remove_resources(self, devices: List[Device], state: TaskState, resources: ResourceSet):
        for device in devices:
            self.remove_device_resources(device, state, resources)

    def check_device_resources(self, device: Device, state: TaskState, resources: ResourceSet) -> bool:
        if device not in self.pool:
            return False

        if state not in self.pool[device]:
            raise ValueError(
                f"Invalid state {state} for Device Resource Request. Valid states are {self.pool[device].keys()}")

        return self.pool[device][state] >= resources

    def check_resources(self, devices: List[Device], state: TaskState, resources: ResourceSet) -> bool:
        for device in devices:
            if not self.check_device_resources(device, state, resources):
                return False
        return True

    def __str__(self) -> str:
        return f"ResourcePool({self.pool})"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, device: Device, state: TaskState = None) -> Dict[TaskState, ResourceSet] | ResourceSet:
        if state is None:
            return self.pool[device]
        else:
            return self.pool[device][state]

    def __setitem__(self, device: Device, state: TaskState, value: Dict[TaskState, ResourceSet]):
        self.pool[device][state] = value

    def __contains__(self, device: Device, state: TaskState) -> bool:
        return device in self.pool and state in self.pool[device]
