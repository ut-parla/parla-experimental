from ..types import Architecture, Device, TaskID, DataID, DataInfo
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict as DefaultDict
from enum import IntEnum


class DataState(IntEnum):
    NOT = -1
    PLANNED = 0
    VALID = 1
    STALE = 2


class DataPhase(IntEnum):
    MAPPED = 1
    RESERVED = 2
    MOVING = 3
    ACTIVE = 4

# TODO(WILL): ENDED HERE Sept. 26, 2022
# Next step: Implement these states in Data and DataPool
#            Make check_resources/reserve_resources in task


@dataclass(slots=True)
class DataStatus():
    mapped_tasks: Set[TaskID] = field(default_factory=set)
    reserved_tasks: Set[TaskID] = field(default_factory=set)
    moving_tasks: Set[TaskID] = field(default_factory=set)
    active_tasks: Set[TaskID] = field(default_factory=set)

    state = DataState.NOT

    mapped_counter: int = 0
    reserved_counter: int = 0
    moving_counter: int = 0
    active_counter: int = 0

    def is_mapped(self) -> bool:
        return self.mapped_counter > 0

    def is_reserved(self) -> bool:
        return self.reserved_counter > 0

    def is_moving(self) -> bool:
        return self.moving_counter > 0

    def is_active(self) -> bool:
        return self.active_counter > 0

    def __str__(self):
        return f"DataStatus({self.stale_counter}, {self.used_counter}, {self.prefetch_counter}, {self.transfer_counter})"

    def __repr__(self):
        return self.__str__()


@dataclass(slots=True)
class SimulatedData:
    name: DataID
    info: DataInfo

    status: Dict[Device, DataStatus] = field(
        default_factory=DefaultDict(lambda x: DataStatus()))

    def __post_init__(self):
        devices = DataInfo.location

        if isinstance(devices, Device):
            devices = (device,)

        for device in devices:
            self.status[device] = DataStatus()

    def acquire(self, devices: List[Device] | Device):

        if isinstance(devices, Device):
            devices = [devices]

        for device in devices:
            if device in self.status:
                if self.status[device].is_stale():
                    raise RuntimeError(
                        "Attempting to acquire stale data: {self.name} on {device}")

                self.status[device].used_counter += 1
                assert (self.status[device].used_counter >= 0)
            else:
                raise RuntimeError(
                    "Attempting to acquire non-existent data: {self.name} on {device}")

    def release(self, devices: List[Device] | Device):

        if isinstance(devices, Device):
            devices = [devices]

        for device in devices:
            if device in self.status:
                if self.status[device].is_stale():
                    raise RuntimeError(
                        "Attempting to release stale data: {self.name} on {device}")

                self.status[device].used_counter -= 1
                assert (self.status[device].used_counter >= 0)
            else:
                raise RuntimeError(
                    "Attempting to release non-existent data: {self.name} on {device}")

    def evict(self, devices: List[Device] | Device):
        if isinstance(devices, Device):
            devices = [devices]

        for device in devices:
            if device in self.status:

                if not self.status.is_stale():
                    raise RuntimeError(
                        f"Attempting to evict non-stale data: {self.name} on {device}")

                elif self.status.is_used():
                    raise RuntimeError(
                        f"Attempting to evict used data: {self.name} on {device}")
                del self.status[device]

                # Device.delete_data should be called here

    def is_valid(self, device: Device, allow_moving=False) -> bool:
        if device in self.status:
            if self.status[device].is_stale():
                return False
            elif not allow_moving and self.status[device].is_moving():
                return False
            else:
                return True
        return False

    def valid_sources(self, allow_moving=False) -> List[Device]:
        sources = []
        for device, status in self.status.items():
            if self.is_valid(device, allow_moving):
                sources.append(device)

        if len(sources) == 0:
            raise RuntimeError(
                "No valid sources found for: {self.name}")
        return sources

    def __str__(self):
        return f"Data({self.name}) | Status: {self.status}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name
