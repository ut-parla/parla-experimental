from .task import *
from .data import *
from .device import *
from .queue import *
from .events import *
from .resources import *
from .task import *
from .topology import *

from ..types import Architecture, Device, TaskID, TaskState, TaskType
from ..types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict

from rich import print


@dataclass(slots=True)
class ObjectRegistry:
    # Object References (Hashable Name -> Object)
    devicemap: Dict[Device, SimulatedDevice] = field(default_factory=dict)
    taskmap: Dict[TaskID, SimulatedTask] = field(default_factory=dict)
    datamap: Dict[DataID, SimulatedData] = field(default_factory=dict)


# TODO: Rename to "SystemState"
@dataclass(slots=True)
class SchedulerState:
    topology: SimulatedTopology = None
    data_pool: DataPool = None
    resource_pool: ResourcePool = None
    objects: ObjectRegistry = None

    def __post_init__(self):

        if self.topology is None:
            raise ValueError("Topology must be specified.")

        self.objects = ObjectRegistry()

        for device in self.topology.devices:
            self.objects.devicemap[device.name] = device

        self.data_pool = DataPool()
        self.resource_pool = ResourcePool(devices=self.topology.devices)


class SchedulerType(IntEnum):
    PARLA = 0


class SchedulerArchitecture:
    completed_tasks: List[TaskID] = []

    def __getitem__(self, func):
        try:
            function = getattr(self, func)
        except AttributeError:
            raise NotImplementedError(
                f"SchedulerArchitecture does not have function {event.func}")
        return function

    def __post_init__(self, topology: SimulatedTopology = None):
        pass

    def initial_events(self, scheduler_state: SchedulerState, max_tasks: int = None) -> List[Event]:
        """
        This event adds initial events to an event queue.
        """
        pass

    def mapper(self, scheduler_state: SchedulerState, max_tasks: int = None) -> List[Event]:
        pass

    def reserver(self, scheduler_state: SchedulerState, max_tasks: int = None) -> List[Event]:
        pass

    def launcher(self, scheduler_state: SchedulerState, max_tasks: int = None) -> List[Event]:
        pass

    def complete_task(self, scheduler_state: SchedulerState, task_id: TaskID) -> List[Event]:
        pass

    def __str__(self):
        return f"SchedulerArchitecture()"

    def __repr__(self):
        self.__str__()

    def __post_init__(self, topology=None):
        pass


@dataclass(slots=True)
class ParlaArchitecture(SchedulerArchitecture):
    topology: SimulatedTopology = None
    spawned_tasks: TaskQueue = TaskQueue()
    mappable_tasks: TaskQueue = TaskQueue()
    mapped_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)
    reservable_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)
    reserved_tasks: Dict[Device, Dict[TaskType, TaskQueue]] = field(default_factory=dict)
    launchable_tasks: Dict[Device, Dict[TaskType, TaskQueue]] = field(default_factory=dict)
    launched_tasks:  Dict[Device, TaskQueue] = field(default_factory=dict)

    def __post_init__(self):

        if self.topology is None:
            raise ValueError("Topology must be specified.")

        for device in self.topology.devices:

            self.mapped_tasks[device.name] = TaskQueue()
            self.reservable_tasks[device.name] = TaskQueue()

            self.reserved_tasks[device.name] = dict()
            self.reserved_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.reserved_tasks[device.name][TaskType.COMPUTE] = TaskQueue()

            self.launchable_tasks[device.name] = dict()
            self.launchable_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.launchable_tasks[device.name][TaskType.COMPUTE] = TaskQueue()


    def initial_events(self, scheduler_state: SchedulerState, max_tasks: int = None) -> List[Event]:
        print("Initial events are invoked")
        return []


    def launcher(self, scheduler_state: SchedulerState, event: Launcher) -> List[Tuple[float, Event]]:
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

    def complete_task(self, scheduler_state: SchedulerState, event: TaskCompleted) -> List[Tuple[float, Event]]:
        task_id = event.task_id

        if task_id is None or task_id not in self.taskmap:
            raise RuntimeError(f"Task {task_id} does not exist!")

        task = self.taskmap[task_id]

        if task is None:
            raise RuntimeError(f"Task {task_id} does not exist!")

        # Stop reserving memory
        # recent_task.free_resources(self.resource_pool)

        # Remove from active device queues
        # for device in recent_task.locations:
        #       self.devicespace[device].pop_local_active_task()

        # Update dependencies
        # recent_task.finish()
        return []

    def map_task(self, scheduler_state: SchedulerState, event: Mapper) -> List[Tuple[float, Event]]:
        return []

    def reserve_task(self, scheduler_state: SchedulerState, event: Reserver) -> List[Tuple[float, Event]]:
        return []


@dataclass(slots=True)
class SimulatedScheduler:
    topology: SimulatedTopology
    task_list: List[SimulatedTask]
    name: str = "SimulatedScheduler"
    # Task Queues for Runtime Phases
    mechanisms: SchedulerArchitecture = None
    state: SchedulerState = None

    events: EventQueue = EventQueue()
    time: float = 0.0

    def __post_init__(self, scheduler: SchedulerType = SchedulerType.PARLA):

        self.state = SchedulerState(topology=self.topology)

        if scheduler == SchedulerType.PARLA:
            self.mechanisms = ParlaArchitecture(topology=self.topology)
        else:
            raise NotImplementedError(
                f"Scheduler type {scheduler} is not implemented")

    def __str__(self):
        return f"Scheduler {self.name} | Current Time: {self.time}"

    def process_event(self, event: Event):
        # New events are created from the current event.
        new_event_pairs = self.mechanisms[event.func](self.state, event)
        # Append new events and their completion times to the event queue
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

    def run(self):
        initial_event = Event("initial_events")
        self.events.put(initial_event, 0)

        for event_pair in GetNextEvent(self.events):
            if event_pair:
                completion_time, event = event_pair
                # Process Event
                new_events = self.process_event(event)
                # Advance time
                self.time = max(self.time, completion_time)
                # Update Log
