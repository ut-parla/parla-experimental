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

from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict

from .scheduler import SchedulerArchitecture, SystemState, SchedulerOptions

from rich import print


@SchedulerOptions.register_scheduler("parla")
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

    def initialize(self, tasks: List[SimulatedTask]) -> List[EventPair]:
        # Initialize the set of visible tasks
        self.add_initial_tasks(tasks)

        # Initialize the event queue
        next_event = Mapper()
        next_time = Time(0)
        return [(next_time, next_event)]

    def add_initial_tasks(self, tasks: List[SimulatedTask]):
        """
        Append an initial task who does not have any dependency to
        a spawned task queue.
        """
        for task in tasks:
            self.spawned_tasks.put(task)

    def launcher(
        self, scheduler_state: SystemState, event: Launcher
    ) -> List[EventPair]:
        return []

    def complete_task(
        self, scheduler_state: SystemState, event: TaskCompleted
    ) -> List[EventPair]:
        return []

    def map_task(self, scheduler_state: SystemState, event: Mapper) -> List[EventPair]:
        return []

    def reserve_task(
        self, scheduler_state: SystemState, event: Reserver
    ) -> List[EventPair]:
        return []
