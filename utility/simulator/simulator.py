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

from .schedulers import *

#from rich import print

from .rl.models.model import *
from .rl.models.env import *


@dataclass(slots=True)
class SimulatedScheduler:
    topology: InitVar[SimulatedTopology]
    scheduler_type: InitVar[str] = "parla"
    rl_mapper: InitVar[RLModel] = None
    rl_environment: InitVar[ParlaRLEnvironment] = None
    tasks: List[TaskID] = field(default_factory=list)
    name: str = "SimulatedScheduler"
    mechanisms: SchedulerArchitecture = field(init=False)
    state: SystemState = field(init=False)
    log_level: int = 0

    events: EventQueue = EventQueue()

    def __post_init__(self, topology: SimulatedTopology, scheduler_type: str = "parla", rl_mapper: RLModel = None, rl_environment: ParlaRLEnvironment = None):
        self.state = SystemState(topology=topology)
        scheduler_arch = SchedulerOptions.get_scheduler(scheduler_type)
        print(f"Scheduler Architecture: {scheduler_arch}")
        self.mechanisms = scheduler_arch(topology=topology, rl_mapper=rl_mapper, rl_environment=rl_environment)

    def __str__(self):
        return f"Scheduler {self.name} | Current Time: {self.time}"

    @property
    def time(self):
        return self.state.time

    @time.setter
    def time(self, time):
        self.state.time = time

    def register_taskmap(self, taskmap: SimulatedTaskMap):
        self.state.register_tasks(taskmap)

    def register_datamap(self, datamap: SimulatedDataMap):
        self.state.register_data(datamap)

    def add_initial_tasks(self, tasks: List[TaskID]):
        self.tasks.extend(tasks)

    def __repr__(self):
        return self.__str__()

    def __rich_repr__(self):
        yield "name", self.name
        yield "time", self.time
        yield "architecture", self.mechanisms
        yield "events", self.events

    def record(self):
        pass

    def process_event(self, event: Event):
        # New events are created from the current event.
        new_event_pairs = self.mechanisms[event](self.state)

        # Append new events and their completion times to the event queue
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

    def run(self):
        new_event_pairs = self.mechanisms.initialize(self.tasks, self.state)
        for completion_time, new_event in new_event_pairs:
            self.events.put(new_event, completion_time)

        from rich import print

        event_count = 0

        next_events = EventIterator(self.events, peek=False)
        for event_pair in next_events:
            if event_pair:
                event_count += 1
                completion_time, event = event_pair
                print(f"Event: {event} at {completion_time}")

                # Advance time
                self.time = max(self.time, completion_time)
                # print("State", self.mechanisms)

                # Process Event
                new_events = self.process_event(event)
                # Update Log
                self.record()

        print(f"Event Count: {event_count}")
        print(f"Execution time: {self.time}")
        # print(self.mechanisms)
