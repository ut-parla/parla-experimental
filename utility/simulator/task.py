from __future__ import annotations
from ..types import TaskID, TaskInfo, TaskState, TaskStatus, DataAccess, Time, TaskType

from ..types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap
from ..types import Architecture, Device
from ..types import DataInfo
from typing import List, Dict, Set, Tuple, Optional, Self, Sequence

from .queue import PriorityQueue
from dataclasses import dataclass, field

from .resources import ResourcePool
from .device import ResourceSet
from .datapool import DataPool


@dataclass(slots=True)
class TaskTimes:
    duration: Time = field(default_factory=Time)
    state_times: Dict[TaskState, Time] = field(default_factory=dict)
    status_times: Dict[TaskStatus, Time] = field(default_factory=dict)

    def __post_init__(self):
        self.state_times = {}
        self.status_times = {}

        for state in TaskState:
            self.state_times[state] = Time(0)
        for status in TaskStatus:
            self.status_times[status] = Time(0)

    def __getitem__(self, state: TaskState | TaskStatus) -> Time:
        if isinstance(state, TaskState):
            return self.state_times[state]
        else:
            return self.status_times[state]

    def __setitem__(self, state: TaskState | TaskStatus, time: Time):
        if isinstance(state, TaskState):
            self.state_times[state] = time
        else:
            self.status_times[state] = time


@dataclass(slots=True, init=False)
class TaskCounters:
    remaining_deps_states: Dict[TaskState, int] = field(default_factory=dict)
    remaining_deps_status: Dict[TaskStatus, int] = field(default_factory=dict)

    def __init__(self, info: TaskInfo):
        self.remaining_deps_states = {}
        self.remaining_deps_status = {}

        for state in TaskState:
            self.remaining_deps_states[state] = len(info.dependencies)

        for state in TaskStatus:
            self.remaining_deps_status[state] = len(info.dependencies)

    def __str__(self) -> str:
        return f"TaskCounters({self.remaining_deps_states})"

    def __repr__(self) -> str:
        return self.__str__()

    def check_count(self, key: TaskStatus | TaskState) -> bool:
        if isinstance(key, TaskState):
            value = self.remaining_deps_states[key]
            assert value >= 0
            return value == 0
        else:
            value = self.remaining_deps_status[key]
            assert value >= 0
            return value == 0

    def notified_state(self, new_state: TaskState) -> Optional[TaskStatus]:
        self.remaining_deps_states[new_state] -= 1
        if self.check_count(new_state):
            if new_status := TaskState.matching_status(new_state):
                return new_status

    def notified_status(self, new_status: TaskStatus) -> None:
        self.remaining_deps_status[new_status] -= 1


@dataclass(slots=True)
class SimulatedTask:
    name: TaskID
    info: TaskInfo
    state: TaskState = TaskState.SPAWNED
    status: Set[TaskStatus] = field(default_factory=set)
    times: TaskTimes = field(default_factory=TaskTimes)
    counters: TaskCounters = field(init=False)
    dependents: List[TaskID] = field(default_factory=list)
    resources: List[ResourceSet] = field(default_factory=list)

    def __post_init__(self):
        self.counters = TaskCounters(self.info)

    def set_status(self, new_status: TaskStatus, time: Time):
        # print(f"Setting {self.name} to {new_status}. Status: {self.status}")
        self.times[new_status] = time
        self.status.add(new_status)

    def set_state(self, new_state: TaskState, time: Time):
        # print(f"Setting {self.name} to {new_state}. State: {self.state}")

        TaskStatus.check_valid_transition(self.status, new_state)
        TaskState.check_valid_transition(self.state, new_state)

        self.times[new_state] = time
        self.state = new_state

    def set_states(self, new_states: List[TaskState], time: Time):
        for state in new_states:
            self.set_state(state, time)

    @property
    def priority(self) -> int:
        return self.info.order

    @priority.setter
    def priority(self, priority: int):
        self.info.order = priority

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
    def assigned_devices(self) -> Optional[Tuple[Device, ...]]:
        if isinstance(self.info.mapping, Device):
            return (self.info.mapping,)
        else:
            return self.info.mapping

    @assigned_devices.setter
    def assigned_devices(self, devices: Tuple[Device, ...]):
        # print(f"Setting {self.name} to {devices}")
        self.info.mapping = devices

    @property
    def read_data_list(self) -> List[DataAccess]:
        return self.info.data_dependencies.read

    @property
    def write_data_list(self) -> List[DataAccess]:
        return self.info.data_dependencies.write

    def add_dependency(
        self,
        task: TaskID,
        states: List[TaskState] = [],
        statuses: List[TaskStatus] = [],
    ):
        self.info.dependencies.append(task)
        for state in states:
            self.counters.remaining_deps_states[state] += 1
        for status in statuses:
            self.counters.remaining_deps_status[status] += 1

    def add_dependent(self, task: TaskID):
        self.dependents.append(task)

    def notify_state(self, state: TaskState, taskmap: SimulatedTaskMap, time: Time):
        # Notify only if changed
        # print(f"Task {self.name} in state {self.state}. Notifying {state}")
        if self.state == state:
            # print(f"Task {self.name} already in state {state}")
            return

        for taskid in self.dependents:
            task = taskmap[taskid]
            if new_status := task.counters.notified_state(state):
                task.notify_status(new_status, taskmap, time)

        self.set_state(state, time)

    def notify_status(self, status: TaskStatus, taskmap: SimulatedTaskMap, time: Time):
        for taskid in self.dependents:
            task = taskmap[taskid]
            task.counters.notified_status(status)

        self.set_status(status, time)

    def check_status(
        self, status: TaskStatus, taskmap: SimulatedTaskMap, time: Time
    ) -> bool:
        if checked_state := TaskStatus.matching_state(status):
            if status not in self.status and self.counters.check_count(checked_state):
                self.notify_status(status, taskmap, time)
                return True
        return status in self.status

    def set_resources(self, devices: Device | Tuple[Device, ...]):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"Task({self.name}, {self.state})"

    def __rich_repr__(self):
        yield "name", self.name
        yield "state", self.state
        yield "status", self.status
        yield "duration", self.duration
        yield "dependencies", self.dependencies
        yield "assigned_devices", self.assigned_devices

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

    @property
    def type(self):
        raise NotImplementedError


@dataclass(slots=True)
class SimulatedComputeTask(SimulatedTask):
    datatasks: List[Self] = field(default_factory=list)
    type: TaskType = TaskType.COMPUTE

    def add_data_dependency(self, task: TaskID):
        self.add_dependency(task, states=[TaskState.COMPLETED])

    def set_duration(
        self, device: Device | Tuple[Device, ...], system_state: "SystemState"
    ):
        runtime_infos = self.get_runtime_info(device)
        max_time = max([runtime_info.task_time for runtime_info in runtime_infos])
        self.duration = Time(max_time)

    def set_resources(
        self, devices: Device | Tuple[Device, ...], data_inputs: bool = False
    ):
        if isinstance(devices, Device):
            devices = (devices,)
        runtime_info_list = self.get_runtime_info(devices)
        for runtime_info in runtime_info_list:
            vcus = runtime_info.device_fraction
            memory = runtime_info.memory
            self.resources.append(ResourceSet(vcus=vcus, memory=memory, copy=0))


@dataclass(slots=True)
class SimulatedDataTask(SimulatedTask):
    type: TaskType = TaskType.DATA

    def set_duration(
        self, device: Device | Tuple[Device, ...], system_state: "SystemState"
    ):
        # Data movement tasks are single device
        assert isinstance(device, Device)

        # TODO: This is the data movement time
        raise NotImplementedError("TODO: implement set_duration for SimulatedDataTask")

    def set_resources(
        self, devices: Device | Tuple[Device, ...], data_inputs: bool = False
    ):
        raise NotImplementedError("TODO: implement set_resources for SimulatedDataTask")


SimulatedTaskMap = Dict[
    TaskID, SimulatedTask | SimulatedComputeTask | SimulatedDataTask
]
SimulatedComputeTaskMap = Dict[TaskID, SimulatedComputeTask]
SimulatedDataTaskMap = Dict[TaskID, SimulatedDataTask]
