from utility.simulator.data import List
from utility.simulator.device import EventPair, List, SimulatedTask
from utility.simulator.queue import EventPair, SimulatedTask
from utility.simulator.resources import List
from utility.simulator.topology import List
from ..task import List, SimulatedTask, SimulatedDataTask, SimulatedComputeTask
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


@SchedulerOptions.register_scheduler("minimal")
@dataclass(slots=True)
class MinimalArchitecture(SchedulerArchitecture):
    topology: SimulatedTopology

    spawned_tasks: TaskQueue = TaskQueue()
    launchable_tasks: Dict[Device, Dict[TaskType, TaskQueue]] = field(
        default_factory=dict
    )
    launched_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)

    def __post_init__(self):
        assert self.topology is not None

        for device in self.topology.devices:
            self.launchable_tasks[device.name] = dict()
            self.launchable_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.launchable_tasks[device.name][TaskType.COMPUTE] = TaskQueue()

    def initialize(self, tasks: List[SimulatedTask]) -> List[EventPair]:
        # Initialize the set of visible tasks
        self.add_initial_tasks(tasks)

        # Initialize the event queue
        next_event = Launcher()
        next_time = Time(0)
        return [(next_time, next_event)]

    def add_initial_tasks(self, tasks: List[SimulatedTask]):
        for task in tasks:
            self.spawned_tasks.put(task)

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
