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

    events: EventQueue = EventQueue()
    time: int = 0

    def __post_init__(self, scheduler_type: str = "parla"):
        self.state = SystemState(topology=self.topology)
        self.mechanisms = SchedulerOptions.get_scheduler(scheduler_type)

    def __str__(self):
        return f"Scheduler {self.name} | Current Time: {self.time}"

    def register_tasks(self, taskmap: SimulatedTaskMap):
        self.state.register_tasks(taskmap)

    def register_data(self, datamap: SimulatedDataMap):
        self.state.register_data(datamap)

    def process_event(self, event: Event):
        # New events are created from the current event.
        new_event_pairs = self.mechanisms[event](self.state)

        # Append new events and their completion times to the event queue
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

    def run(self):
        new_event_pairs = self.mechanisms.initialize()
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
