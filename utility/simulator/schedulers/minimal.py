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
    topology: InitVar[SimulatedTopology]

    spawned_tasks: TaskQueue = TaskQueue()
    launchable_tasks: Dict[Device, Dict[TaskType, TaskQueue]] = field(
        default_factory=dict
    )
    launched_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)

    def __post_init__(self, topology: SimulatedTopology):
        assert topology is not None

        for device in topology.devices:
            self.launchable_tasks[device.name] = dict()
            self.launchable_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.launchable_tasks[device.name][TaskType.COMPUTE] = TaskQueue()

            self.launched_tasks[device.name] = TaskQueue()

    def initialize(
        self, tasks: List[TaskID], scheduler_state: SystemState
    ) -> List[EventPair]:
        objects = scheduler_state.objects
        assert objects is not None

        task_objects = [objects.get_task(task) for task in tasks]

        # Initialize the set of visible tasks
        self.add_initial_tasks(task_objects)

        # Initialize the event queue
        next_event = Mapper()
        next_time = Time(0)
        return [(next_time, next_event)]

    def add_initial_tasks(self, tasks: List[SimulatedTask]):
        print("Adding initial tasks", tasks)
        for task in tasks:
            # task.check_and_set_state(TaskState.MAPPED, Time(0))
            self.spawned_tasks.put(task)

    def mapper(self, scheduler_state: SystemState, event: Mapper) -> List[EventPair]:
        print("Mapping tasks...")
        next_tasks = TaskIterator(self.spawned_tasks)

        current_time = scheduler_state.time
        assert current_time is not None

        objects = scheduler_state.objects
        assert objects is not None

        for priority, taskid in next_tasks:
            print(next_tasks)
            print(taskid)
            task = objects.get_task(taskid)
            assert task is not None

            task.check_and_set_state(TaskState.MAPPED, TaskState.MAPPABLE, current_time)

            # Move task to mapped queue
            devices = task.assigned_devices
            if devices is None:
                raise ValueError(
                    f"Task {task.name} has no assigned devices. Minimal scheduler requires that all tasks have an assigned device at spawn."
                )

            device = devices[0]
            self.launchable_tasks[device][task.type].put_id(taskid, priority=priority)
            next_tasks.success()
            task.notify(TaskState.MAPPED, objects.taskmap, current_time)

        launcher_pair = (current_time, Launcher())
        return [launcher_pair]

    def launcher(
        self, scheduler_state: SystemState, event: Launcher
    ) -> List[EventPair]:
        print("Launching tasks...")
        objects = scheduler_state.objects
        assert objects is not None

        current_time = scheduler_state.time
        assert current_time is not None

        # Launch tasks up to max_tasks or resource limits

        next_tasks = MultiTaskIterator(self.launchable_tasks)
        next_events = []
        for priority, taskid in next_tasks:
            print(self)
            print(f"Launching task: {taskid}")
            task = objects.get_task(taskid)
            assert task is not None
            print(task)

            # Launch task
            task.check_and_set_state(
                TaskState.RESERVED, TaskState.RESERVABLE, current_time
            )
            task.notify(TaskState.RESERVED, objects.taskmap, current_time)
            task.check_and_set_state(
                TaskState.COMPLETED, TaskState.LAUNCHABLE, current_time
            )
            if task.state == TaskState.LAUNCHABLE:
                if task.assigned_devices is None:
                    raise ValueError("Task has no assigned devices.")

                device = task.assigned_devices[0]
                task.set_duration(device, scheduler_state)
                completion_time = current_time + task.duration
                self.launched_tasks[device].put_id(taskid, completion_time)

                # Create completion event
                completion_event = TaskCompleted(task=taskid)
                next_events.append((completion_time, completion_event))

                next_tasks.success()
                task.notify(TaskState.LAUNCHED, objects.taskmap, current_time)
            else:
                next_tasks.fail()

        return next_events
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
        print(f"Completing task: {event.task}...")
        objects = scheduler_state.objects
        assert objects is not None
        taskid = event.task
        task = objects.get_task(taskid)

        # Stop reserving memory
        # recent_task.free_resources(self.resource_pool)

        # Remove task from launched queues
        devices = task.assigned_devices
        if devices is None:
            raise ValueError(f"Task {task.name} has no assigned devices.")
        device = devices[0]

        expected_completion_time, task_at_head = self.launched_tasks[device].peek()
        if task_at_head == taskid:
            self.launched_tasks[device].get()
        else:
            raise ValueError(
                f"Invalid state: Task {task.name} is not at the head of the launched queue."
            )

        # Update dependencies
        task.notify(TaskState.COMPLETED, objects.taskmap, scheduler_state.time)

        return []
