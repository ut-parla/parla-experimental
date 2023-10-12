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

from schedulers.scheduler import SchedulerArchitecture, SystemState, SchedulerOptions

from rich import print


@dataclass(slots=True)
class SimulatedScheduler:
    topology: SimulatedTopology
    tasks: List[SimulatedTask] = field(default_factory=list)
    name: str = "SimulatedScheduler"
    mechanisms: SchedulerArchitecture = field(init=False)
    state: SystemState = field(init=False)
    log_level: int = 0

    events: EventQueue = EventQueue()
    time: Time = field(default_factory=Time)

    def __post_init__(self, scheduler_type: str = "parla"):
        self.state = SystemState(topology=self.topology)
        self.mechanisms = SchedulerOptions.get_scheduler(scheduler_type)

    def __str__(self):
        return f"Scheduler {self.name} | Current Time: {self.time}"

    def register_taskmap(self, taskmap: SimulatedTaskMap):
        self.state.register_tasks(taskmap)

    def register_datamap(self, datamap: SimulatedDataMap):
        self.state.register_data(datamap)

    def add_initial_tasks(self, tasks: List[SimulatedTask]):
        self.tasks.extend(tasks)

    def record():
        pass

    def process_event(self, event: Event):
        # New events are created from the current event.
        new_event_pairs = self.mechanisms[event](self.state)

        # Append new events and their completion times to the event queue
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

    def run(self):
        new_event_pairs = self.mechanisms.initialize(self.tasks)
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

        next_events = EventIterator(self.events)
        for event_pair in next_events:
            if event_pair:
                completion_time, event = event_pair
                # Process Event
                new_events = self.process_event(event)
                # Advance time
                self.time = max(self.time, completion_time)
                # Update Log
                self.record()
