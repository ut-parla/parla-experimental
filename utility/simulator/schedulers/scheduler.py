from ..task import SimulatedTask, SimulatedDataTask, SimulatedComputeTask
from ..data import *
from ..device import *
from ..queue import *
from ..events import *
from ..resources import *
from ..task import *
from ..topology import *

from ...types import Architecture, Device, TaskID, TaskState, TaskType, Time
from ...types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap

from typing import List, Dict, Set, Tuple, Optional, Callable, Type, Sequence
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict

from rich import print


@dataclass(slots=True)
class ObjectRegistry:
    # Object References (Hashable Name -> Object)
    devicemap: Dict[Device, SimulatedDevice] = field(default_factory=dict)
    taskmap: SimulatedTaskMap = field(default_factory=dict)
    datamap: Dict[DataID, SimulatedData] = field(default_factory=dict)

    def add_task(self, task: SimulatedTask):
        self.taskmap[task.name] = task

    def get_task(self, task_id: Optional[TaskID]) -> SimulatedTask:
        assert task_id is not None
        assert self.taskmap is not None

        if task_id not in self.taskmap:
            raise ValueError(
                f"System state does not have a reference to task: {task_id}."
            )

        task = self.taskmap[task_id]

        if task is None:
            raise ValueError(
                f"System state has a reference to task {task_id} but it is None."
            )

        return task

    def add_data(self, data: SimulatedData):
        self.datamap[data.name] = data

    def get_data(self, data_id: Optional[DataID]) -> SimulatedData:
        assert data_id is not None
        assert self.datamap is not None

        if data_id not in self.datamap:
            raise ValueError(
                f"System state does not have a reference to data: {data_id}."
            )

        data = self.datamap[data_id]

        if data is None:
            raise ValueError(
                f"System state has a reference to data {data_id} but it is None."
            )

        return data

    def add_device(self, device: SimulatedDevice):
        self.devicemap[device.name] = device

    def get_device(self, device_id: Optional[Device]) -> SimulatedDevice:
        assert device_id is not None
        assert self.devicemap is not None

        if device_id not in self.devicemap:
            raise ValueError(
                f"System state does not have a reference to device: {device_id}."
            )

        device = self.devicemap[device_id]

        if device is None:
            raise ValueError(
                f"System state has a reference to device {device_id} but it is None."
            )

        return device


@dataclass(slots=True)
class SystemState:
    topology: SimulatedTopology
    data_pool: DataPool = field(init=False)
    resource_pool: ResourcePool = field(init=False)
    objects: ObjectRegistry = field(init=False)
    time: Time = field(default_factory=Time)

    def __post_init__(self):
        assert self.topology is not None

        self.objects = ObjectRegistry()

        for device in self.topology.devices:
            self.objects.add_device(device)

        self.data_pool = DataPool(devices=self.topology.devices)
        self.resource_pool = ResourcePool(devices=self.topology.devices)

    def register_tasks(self, taskmap: SimulatedTaskMap, copy: bool = False):
        if copy:
            self.objects.taskmap = taskmap.copy()
        else:
            self.objects.taskmap = taskmap

    def register_data(self, datamap: Dict[DataID, SimulatedData], copy: bool = False):
        if copy:
            self.objects.datamap = datamap.copy()
        else:
            self.objects.datamap = datamap

    def register_devices(
        self, devicemap: Dict[Device, SimulatedDevice], copy: bool = False
    ):
        if copy:
            self.objects.devicemap = devicemap.copy()
        else:
            self.objects.devicemap = devicemap

    def check_resources(self, taskid: TaskID, state: TaskState) -> bool:
        # Check that the resources are available
        raise NotImplementedError()

    def acquire_resources(self, taskid: TaskID, state: TaskState):
        # Reserve the resources
        raise NotImplementedError()

    def release_resources(self, taskid: TaskID, state: TaskState):
        # Release the resources
        raise NotImplementedError()

    def use_data(
        self, taskid: TaskID, state: TaskState, data: DataID, access: AccessType
    ):
        # Update data tracking
        raise NotImplementedError()

    def release_data(
        self, taskid: TaskID, state: TaskState, data: DataID, access: AccessType
    ):
        # Update data tracking
        raise NotImplementedError()


@dataclass(slots=True)
class SchedulerArchitecture:
    topology: InitVar[SimulatedTopology]
    completed_tasks: List[TaskID] = field(default_factory=list)

    def __post_init__(self, topology: SimulatedTopology):
        assert topology is not None

    def __getitem__(self, event: Event) -> Callable[[SystemState], Sequence[EventPair]]:
        try:
            function = getattr(self, event.func)
        except AttributeError:
            raise NotImplementedError(
                f"SchedulerArchitecture does not implement function {event.func} for event {event}."
            )

        def wrapper(scheduler_state: SystemState) -> Sequence[EventPair]:
            return function(scheduler_state, event)

        return wrapper

    def initialize(
        self, tasks: List[TaskID], scheduler_state: SystemState
    ) -> Sequence[EventPair]:
        raise NotImplementedError()
        return []

    def add_initial_tasks(self, task: SimulatedTask):
        pass

    def mapper(self, scheduler_state: SystemState, event: Event) -> Sequence[EventPair]:
        raise NotImplementedError()
        return []

    def reserver(
        self, scheduler_state: SystemState, event: Event
    ) -> Sequence[EventPair]:
        raise NotImplementedError()
        return []

    def launcher(
        self, scheduler_state: SystemState, event: Event
    ) -> Sequence[EventPair]:
        raise NotImplementedError()
        return []

    def complete_task(
        self, scheduler_state: SystemState, event: Event
    ) -> Sequence[EventPair]:
        return []

    def __str__(self):
        return f"SchedulerArchitecture()"

    def __repr__(self):
        self.__str__()


class SchedulerOptions:
    scheduler_map: Dict[str, Type[SchedulerArchitecture]] = dict()

    @staticmethod
    def register_scheduler(scheduler_type: str) -> Callable[[Type], Type]:
        def decorator(cls):
            if scheduler_type in SchedulerOptions.scheduler_map:
                raise ValueError(
                    f"Scheduler type {scheduler_type} is already registered."
                )
            SchedulerOptions.scheduler_map[scheduler_type] = cls
            return cls

        return decorator

    @staticmethod
    def get_scheduler(scheduler_type: str) -> Type[SchedulerArchitecture]:
        if scheduler_type not in SchedulerOptions.scheduler_map:
            raise ValueError(
                f"Scheduler type `{scheduler_type}` is not registered. Registered types are: {list(SchedulerOptions.scheduler_map.keys())}"
            )
        return SchedulerOptions.scheduler_map[scheduler_type]
