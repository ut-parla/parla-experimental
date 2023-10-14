from __future__ import annotations
from ..types import TaskID, TaskInfo, TaskState, DataAccess, Time

from ..types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap
from ..types import Architecture, Device
from ..types import DataInfo
from typing import List, Dict, Set, Tuple, Optional, Self

from .queue import PriorityQueue
from dataclasses import dataclass, field

from .resources import ResourcePool
from .datapool import DataPool


@dataclass(slots=True)
class TaskTimes:
    duration: Time = field(default_factory=Time)
    transitions: Dict[TaskState, Time] = field(default_factory=dict)

    def __post_init__(self):
        self.transitions = {state: Time(0) for state in TaskState}

    def __getitem__(self, state: TaskState) -> Time:
        return self.transitions[state]

    def __setitem__(self, state: TaskState, time: Time):
        self.transitions[state] = time


@dataclass(slots=True, init=False)
class TaskCounters:
    remaining_deps: Dict[TaskState, int] = field(default_factory=dict)

    def __init__(self, info: TaskInfo):
        self.remaining_deps = {}
        for state in TaskState:
            self.remaining_deps[state] = len(info.dependencies)

    def __str__(self) -> str:
        return f"TaskCounters({self.remaining_deps})"

    def __repr__(self) -> str:
        return self.__str__()

    def _can_transition(self, state: TaskState) -> bool:
        return self.remaining_deps[state] == 0

    def resolve_transition(self, state: TaskState) -> Optional[TaskState]:
        if self._can_transition(state):
            return TaskState.resolve_state_trigger(state)
        else:
            return None


@dataclass(slots=True)
class SimulatedTask:
    name: TaskID
    info: TaskInfo
    state: TaskState = TaskState.SPAWNED
    times: TaskTimes = field(default_factory=TaskTimes)
    counters: TaskCounters = field(init=False)
    dependents: List[TaskID] = field(default_factory=list)

    def __post_init__(self):
        self.counters = TaskCounters(self.info)

    def set_state(self, new_state: TaskState, time: Time):
        TaskState.check_valid_transition(self.state, new_state)

        self.times[new_state] = time
        self.state = new_state

    def set_states(self, new_states: List[TaskState], time: Time):
        for state in new_states:
            self.set_state(state, time)

    @property
    def duration(self) -> Time:
        return self.times.duration

    @duration.setter
    def duration(self, time: Time):
        self.times.duration = time

    @property
    def dependencies(self) -> List[TaskID]:
        return self.info.dependencies

    @dependencies.setter
    def dependencies(self, deps: List[TaskID]):
        self.info.dependencies = deps

    @property
    def assigned_devices(self) -> Optional[Tuple[Device]]:
        if isinstance(self.info.mapping, Device):
            return (self.info.mapping,)
        else:
            self.info.mapping

    @assigned_devices.setter
    def assigned_devices(self, devices: Tuple[Device]):
        self.info.mapping = devices

    @property
    def read_data_list(self) -> List[DataAccess]:
        return self.info.data_dependencies.read

    @property
    def write_data_list(self) -> List[DataAccess]:
        return self.info.data_dependencies.write

    def add_dependency(self, task: TaskID, states: List[TaskState] = []):
        self.info.dependencies.append(task)
        for state in states:
            self.counters.remaining_deps[state] += 1

    def add_dependent(self, task: TaskID):
        self.dependents.append(task)

    def notify(self, state: TaskState, taskmap: SimulatedTaskMap, time: Time):
        for taskid in self.dependents:
            task = taskmap[taskid]
            task.counters.remaining_deps[state] -= 1
            if new_state := task.counters.resolve_transition(state):
                task.set_state(new_state, time)

    def __str__(self) -> str:
        return f"Task({self.name}, {self.state})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Self) -> bool:
        return self.name == other.name

    def get_runtime_info(
        self, device: Device | Tuple[Device, ...]
    ) -> List[TaskRuntimeInfo]:
        return self.info.runtime[device]

    def set_duration(
        self, device: Device | Tuple[Device, ...], system_state: "SystemState"
    ):
        raise NotImplementedError


@dataclass(slots=True)
class SimulatedComputeTask(SimulatedTask):
    datatasks: List[Self] = field(default_factory=list)

    def add_data_dependency(self, task: TaskID):
        self.add_dependency(task, states=[TaskState.COMPLETED])

    def set_duration(
        self, device: Device | Tuple[Device, ...], system_state: "SystemState"
    ):
        runtime_infos = self.get_runtime_info(device)
        max_time = max([runtime_info.task_time for runtime_info in runtime_infos])
        self.duration = Time(max_time)


@dataclass(slots=True)
class SimulatedDataTask(SimulatedTask):
    def set_duration(
        self, device: Device | Tuple[Device, ...], system_state: "SystemState"
    ):
        # Data movement tasks are single device
        assert isinstance(device, Device)

        # TODO: This is the data movement time
        raise NotImplementedError("TODO: implement set_duration for SimulatedDataTask")


SimulatedTaskMap = Dict[
    TaskID, SimulatedTask | SimulatedComputeTask | SimulatedDataTask
]
SimulatedComputeTaskMap = Dict[TaskID, SimulatedComputeTask]
SimulatedDataTaskMap = Dict[TaskID, SimulatedDataTask]
