from __future__ import annotations
from ..types import TaskID, TaskInfo, TaskState, DataAccess

from ..types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap
from ..types import Architecture, Device
from ..types import DataInfo
from typing import List, Dict, Set, Tuple, Optional, Self

from .queue import PriorityQueue
from dataclasses import dataclass

from .resources import ResourcePool
from .datapool import DataPool


@dataclass(slots=True)
class TaskTimes():
    duration: float = 0.0
    spawn_t: float = 0.0
    map_t: float = 0.0
    reserve_t: float = 0.0
    launch_t: float = 0.0
    complete_t: float = 0.0


@dataclass(slots=True)
class TaskCounters():
    unmapped_deps: int = 0
    unreserved_deps: int = 0
    uncompleted_deps: int = 0

    def __post_init__(self, info: TaskInfo):
        self.unmapped_deps = len(info.dependencies)
        self.unreserved_deps = len(info.dependencies)
        self.uncompleted_deps = len(info.dependencies)

    def __str__(self) -> str:
        return f"TaskCounters({self.unmapped_deps}, {self.unreserved_deps}, {self.uncompleted_deps})"

    def __repr__(self) -> str:
        return self.__str__()

    def is_mappable(self) -> bool:
        return self.unmapped_deps == 0

    def is_reservable(self) -> bool:
        return self.unreserved_deps == 0

    def is_launchable(self) -> bool:
        return self.uncompleted_deps == 0


@dataclass(slots=True)
class SimulatedTask:
    name: TaskID
    info: TaskInfo
    state: TaskState = TaskState.SPAWNED
    times: TaskTimes = TaskTimes()
    counters: TaskCounters | None = None

    def __post_init__(self):
        self.counters = TaskCounters(self.info)

    @property
    def duration(self) -> float:
        return self.times.duration

    @property
    def dependencies(self) -> List[TaskID]:
        return self.info.dependencies

    @dependencies.setter
    def dependencies(self, deps: List[TaskID]):
        self.info.dependencies = deps

    @property
    def mapped_devices(self) -> Tuple[Device]:
        return self.info.mapping

    @mapped_devices.setter
    def mapped_devices(self, devices: Tuple[Device]):
        self.info.mapping = devices

    @property
    def read_data_list(self) -> List[DataAccess]:
        return self.info.read_data

    @property
    def write_data_list(self) -> List[DataAccess]:
        return self.info.write_data

    @property
    def is_mappable(self) -> bool:
        return self.counters.is_mappable()

    @property
    def is_reservable(self) -> bool:
        return self.counters.is_reservable()

    @property
    def is_launchable(self) -> bool:
        return self.counters.is_launchable()

    def __str__(self) -> str:
        return f"Task({self.name}, {self.state})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Self) -> bool:
        return self.name == other.name


@dataclass(slots=True)
class SimulatedComputeTask(SimulatedTask):
    pass


@dataclass(slots=True)
class SimulatedDataTask(SimulatedTask):
    pass
