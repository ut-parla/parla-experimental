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
from dataclasses import dataclass
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
    topology: SimulatedTopology
    data_pool: DataPool = None
    resource_pool: ResourcePool = None
    objects: ObjectRegistry = None

    def __post_init__(self, topology: SimulatedTopology = None):

        if topology is None:
            raise ValueError("Topology must be specified.")

        self.objects = ObjectRegistry()
        self.topology = topology

        for device in topology.devices:
            self.objects.devicemap[device.name] = device

        self.data_pool = DataPool()
        self.resource_pool = ResourcePool(topology.devices)


class SchedulerType(IntEnum):
    PARLA = 0


class SchedulerArchitecture:
    completed_tasks: List[TaskID] = []

    def __getitem__(self, event):
        try:
            function = getattr(self, event.func)
        except AttributeError:
            raise NotImplementedError(
                f"SchedulerArchitecture does not have function {event.func}")
        return function

    def __post_init__(self, topology: SimulatedTopology = None):
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
    spawned_tasks: TaskQueue = TaskQueue()
    mappable_tasks: TaskQueue = TaskQueue()
    mapped_tasks: Dict[Device, TaskQueue] | None = None

    reservable_tasks: Dict[Device, TaskQueue] | None = None
    reserved_tasks: Dict[Device, Dict[TaskType, TaskQueue]] | None = None

    launchable_tasks: Dict[Device, Dict[TaskType, TaskQueue]] | None = None
    launched_tasks:  Dict[Device, TaskQueue] | None = None

    def __post_init__(self, topology: SimulatedTopology = None):

        if topology is None:
            raise ValueError("Topology must be specified.")

        for device in topology.devices:

            self.mapped_tasks[device.name] = TaskQueue()
            self.reservable_tasks[device.name] = TaskQueue()

            self.reserved_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.reserved_tasks[device.name][TaskType.COMPUTE] = TaskQueue()

            self.launchable_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.launchable_tasks[device.name][TaskType.COMPUTE] = TaskQueue()

    def launcher(self, scheduler_state: SchedulerState, event: Launcher) -> List[Event]:
        pass
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

    def complete_task(self, scheduler_state: SchedulerState, event: TaskCompleted) -> List[Event]:
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


@dataclass(slots=True)
class SimulatedScheduler:

    name: str

    # Task Queues for Runtime Phases
    mechanisms: SchedulerArchitecture = None
    state: SchedulerState = None

    events: EventQueue = EventQueue()
    time: float = 0.0

    def __post_init__(self, topology, scheduler: SchedulerType = SchedulerType.PARLA):

        self.state = SchedulerState(topology=topology)

        if scheduler == SchedulerType.PARLA:
            self.mechanisms = ParlaArchitecture(topology=topology)
        else:
            raise NotImplementedError(
                f"Scheduler type {scheduler} is not implemented")

    def __str__(self):
        return f"Scheduler {self.name} | Current Time: {self.time}"

    def process_event(self, event: Event):
        self.mechanisms[event.func](self.state, event)

    def run(self):

        for event_pair in GetNextEvent(self.events):
            if event_pair:
                completion_time, event = event_pair
                # Process Event
                self.process_event(event)
                # Advance time
                self.time = max(self.time, completion_time)
                # Update Log
