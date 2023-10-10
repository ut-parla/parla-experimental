from .task import SimulatedTask, SimulatedDataTask, SimulatedComputeTask
from .data import *
from .device import *
from .queue import *
from .events import *
from .resources import *
from .task import *
from .topology import *

from ..types import Architecture, Device, TaskID, TaskState, TaskType, Time
from ..types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap

from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict

from rich import print


@dataclass(slots=True)
class ObjectRegistry:
    # Object References (Hashable Name -> Object)
    devicemap: Dict[Device, SimulatedDevice] = field(default_factory=dict)
    taskmap: Dict[TaskID, SimulatedTask] = field(default_factory=dict)
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

    def __post_init__(self):
        assert self.topology is not None

        self.objects = ObjectRegistry()

        for device in self.topology.devices:
            self.objects.add_device(device)

        self.data_pool = DataPool()
        self.resource_pool = ResourcePool(devices=self.topology.devices)


class SchedulerType(IntEnum):
    PARLA = 0


class SchedulerArchitecture:
    completed_tasks: List[TaskID] = []

    def __getitem__(self, event: Event) -> Callable[[SystemState], List[EventPair]]:
        try:
            function = getattr(self, event.func)
        except AttributeError:
            raise NotImplementedError(
                f"SchedulerArchitecture does not implement function {event.func} for event {event}."
            )

        def wrapper(scheduler_state: SystemState) -> List[EventPair]:
            return function(scheduler_state, event)

        return wrapper

    def add_initial_task(self, task: SimulatedTask):
        pass

    def mapper(self, scheduler_state: SystemState, event: Event) -> List[EventPair]:
        return []

    def reserver(self, scheduler_state: SystemState, event: Event) -> List[EventPair]:
        return []

    def launcher(self, scheduler_state: SystemState, event: Event) -> List[EventPair]:
        return []

    def complete_task(
        self, scheduler_state: SystemState, event: Event
    ) -> List[EventPair]:
        return []

    def __str__(self):
        return f"SchedulerArchitecture()"

    def __repr__(self):
        self.__str__()

    def __post_init__(self, topology=None):
        pass


@dataclass(slots=True)
class ParlaArchitecture(SchedulerArchitecture):
    topology: SimulatedTopology

    spawned_tasks: TaskQueue = TaskQueue()

    # Mapping Phase
    mappable_tasks: TaskQueue = TaskQueue()
    mapped_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)
    # Reserving Phase
    reservable_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)
    reserved_tasks: Dict[Device, Dict[TaskType, TaskQueue]] = field(
        default_factory=dict
    )
    # Launching Phase
    launchable_tasks: Dict[Device, Dict[TaskType, TaskQueue]] = field(
        default_factory=dict
    )
    launched_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)

    def __post_init__(self):
        assert self.topology is not None

        for device in self.topology.devices:
            self.mapped_tasks[device.name] = TaskQueue()
            self.reservable_tasks[device.name] = TaskQueue()

            self.reserved_tasks[device.name] = dict()
            self.reserved_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.reserved_tasks[device.name][TaskType.COMPUTE] = TaskQueue()

            self.launchable_tasks[device.name] = dict()
            self.launchable_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.launchable_tasks[device.name][TaskType.COMPUTE] = TaskQueue()

    def add_initial_task(self, task: SimulatedTask):
        """
        Append an initial task who does not have any dependency to
        a spawned task queue.
        """
        self.spawned_tasks.put(task.name)

    def launcher(
        self, scheduler_state: SystemState, event: Launcher
    ) -> List[EventPair]:
        return []
        # i = 0
        # max_tasks = event.max_tasks

        # while i < max_tasks:

        #     for device in self.devices:
        #         reserved_tasks = self.reserved_tasks[device]
        #         data_tasks = reserved_tasks[TaskType.DATA]
        #         compute_tasks = reserved_tasks[TaskType.COMPUTE]

        #         active_queue = data_tasks

        #         # Check if there are any tasks that can be launched

        #         task = active_queue.peek()
        #         if task
        #         if self.check_resources(task, self.resource_pool):
        #             task = active_queue.get()
        #             duration = estimate_time(task, self.resource_pool)
        #             start_task(task)

        #         if task:
        #             print("Assigning Task:", task.name, task.dependencies)

        #             # data0 = task.read_data[0]
        #             # print(data0)

        #             # Assign memory
        #             task.reserve_resources(self.resource_pool)

        #             # Compute time to completion
        #             task.estimate_time(self.time, self.resource_pool)
        #             # print("Expected Complete: ", task.completion_time)
        #             # "Start" Task
        #             # 1. Locks data (so it can't be evicted)
        #             # 2. Updates data status (to evict stale data)
        #             task.start()

        #             # Lock used data
        #             # task.lock_data()

        #             # Push task to global active queue
        #             self.active_tasks.put((task.completion_time, task))

        #             # Push task to local active queue (for device)
        #             # NOTE: This is mainly just as a convience for logging
        #             device.push_local_active_task(task)

        #             # data0 = task.read_data[0]
        #             # print(data0)

        #             # Update global state log
        #             self.state.set_task_log(
        #                 task.name, "status", "active")
        #             self.state.set_task_log(
        #                 task.name, "start_time", self.time)
        #         else:
        #             continue

    def complete_task(
        self, scheduler_state: SystemState, event: TaskCompleted
    ) -> List[EventPair]:
        objects = scheduler_state.objects
        assert objects is not None
        task = objects.get_task(event.task)

        # Stop reserving memory
        # recent_task.free_resources(self.resource_pool)

        # Remove from active device queues
        # for device in recent_task.locations:
        #       self.devicespace[device].pop_local_active_task()

        # Update dependencies
        # recent_task.finish()
        return []

    def map_task(self, scheduler_state: SystemState, event: Mapper) -> List[EventPair]:
        return []

    def reserve_task(
        self, scheduler_state: SystemState, event: Reserver
    ) -> List[EventPair]:
        return []


@dataclass(slots=True)
class SimulatedScheduler:
    topology: SimulatedTopology
    tasks: List[SimulatedTask] = field(default_factory=list)
    name: str = "SimulatedScheduler"
    mechanisms: SchedulerArchitecture = field(init=False)
    state: SystemState = field(init=False)

    events: EventQueue = EventQueue()
    time: int = 0

    def __post_init__(self, scheduler: SchedulerType = SchedulerType.PARLA):
        self.state = SystemState(topology=self.topology)

        if scheduler == SchedulerType.PARLA:
            self.mechanisms = ParlaArchitecture(topology=self.topology)
        else:
            raise NotImplementedError(f"Scheduler type {scheduler} is not implemented")

    def __str__(self):
        return f"Scheduler {self.name} | Current Time: {self.time}"

    def populate(self, tasks: Optional[List[SimulatedTask]] = None):
        if tasks is None:
            assert self.tasks is not None
        else:
            self.tasks = tasks

        for task in self.tasks:
            self.state.objects.add_task(task)

        for task in self.tasks:
            self.mechanisms.add_initial_task(task)

    def process_event(self, event: Event):
        # New events are created from the current event.
        new_event_pairs = self.mechanisms[event](self.state)

        # Append new events and their completion times to the event queue
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

    def run(self):
        new_event_pairs = self.mechanisms.initiate()
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

        for event_pair in GetNextEvent(self.events):
            if event_pair:
                completion_time, event = event_pair
                # Process Event
                new_events = self.process_event(event)
                # Advance time
                self.time = max(self.time, completion_time)
                # Update Log
