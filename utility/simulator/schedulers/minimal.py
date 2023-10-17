from utility.simulator.data import List
from utility.simulator.device import Event, EventPair, List
from utility.simulator.events import Event
from utility.simulator.queue import Event, EventPair, length
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

from typing import List, Dict, Set, Tuple, Optional, Callable, Sequence
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
    ) -> Sequence[EventPair]:
        objects = scheduler_state.objects
        assert objects is not None

        task_objects = [objects.get_task(task) for task in tasks]

        # Initialize the set of visible tasks
        self.add_initial_tasks(task_objects, scheduler_state)

        # Initialize the event queue
        next_event = Mapper()
        next_time = Time(0)
        return [(next_time, next_event)]

    def add_initial_tasks(
        self, tasks: Sequence[SimulatedTask], scheduler_state: SystemState
    ):
        print("Spawning initial tasks...", tasks)

        current_time = scheduler_state.time
        assert current_time is not None

        objects = scheduler_state.objects
        assert objects is not None

        for task in tasks:
            # task.check_status(TaskStatus.MAPPABLE, objects.taskmap, current_time)
            self.spawned_tasks.put(task)

    def mapper(
        self, scheduler_state: SystemState, event: Mapper
    ) -> Sequence[EventPair]:
        print("Mapping tasks...")
        next_tasks = TaskIterator(self.spawned_tasks)

        current_time = scheduler_state.time
        assert current_time is not None

        objects = scheduler_state.objects
        assert objects is not None

        for priority, taskid in next_tasks:
            print(next_tasks)
            print(f"Processing task {taskid}")
            task = objects.get_task(taskid)
            assert task is not None

            # Check if task is mappable
            if check_status := task.check_status(
                TaskStatus.MAPPABLE, objects.taskmap, current_time
            ):
                # Map task
                devices = task.assigned_devices
                if devices is None:
                    raise ValueError(
                        f"Task {task.name} has no assigned devices. Minimal scheduler requires that all tasks have an assigned device at spawn."
                    )

                device = devices[0]
                self.launchable_tasks[device][task.type].put_id(
                    taskid, priority=priority
                )
                next_tasks.success()
                task.notify_state(TaskState.MAPPED, objects.taskmap, current_time)

        launcher_pair = (current_time, Launcher())
        return [launcher_pair]

    def launcher(
        self, scheduler_state: SystemState, event: Launcher
    ) -> Sequence[EventPair]:
        print("Launching tasks...")
        objects = scheduler_state.objects
        assert objects is not None

        current_time = scheduler_state.time
        assert current_time is not None

        next_events: Sequence[EventPair] = []
        if remaining_tasks := length(self.launchable_tasks):
            mapping_pair = (current_time + 10, Mapper())
            next_events.append(mapping_pair)

        print(f"Remaining tasks: {remaining_tasks}")

        next_tasks = MultiTaskIterator(self.launchable_tasks)
        for priority, taskid in next_tasks:
            print(self)
            print(f"Launching task: {taskid}")
            task = objects.get_task(taskid)
            assert task is not None
            print(task)

            # Process RESERVABLE state (all tasks should be reservable in minimal scheduler)
            if check_reservable := task.check_status(
                TaskStatus.RESERVABLE, objects.taskmap, current_time
            ):
                print(f"Task {task.name} is reservable.")
                task.notify_state(TaskState.RESERVED, objects.taskmap, current_time)
            else:
                next_tasks.fail()
                continue

            # Process LAUNCHABLE state
            if check_launchable := task.check_status(
                TaskStatus.LAUNCHABLE, objects.taskmap, current_time
            ):
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
                task.notify_state(TaskState.LAUNCHED, objects.taskmap, current_time)
            else:
                next_tasks.fail()
                continue

        return next_events

    def complete_task(
        self, scheduler_state: SystemState, event: TaskCompleted
    ) -> Sequence[EventPair]:
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

        if expected := self.launched_tasks[device].peek():
            expected_time, expected_task = expected
            if expected_task != taskid:
                raise ValueError(
                    f"Invalid state: Task {task.name} is not at the head of the launched queue. Expected: {expected_task}, Actual: {taskid}"
                )
            if expected_time != scheduler_state.time:
                raise ValueError(
                    f"Invalid state: Task {task.name} is not expected to complete at this time. Expected: {expected_time}, Actual: {scheduler_state.time}"
                )
            # Remove task from launched queue (it has completed)
            self.launched_tasks[device].get()
        else:
            raise ValueError(
                f"Invalid state: Launch queue for device {device} is empty."
            )

        # Update status of dependencies
        task.notify_state(TaskState.COMPLETED, objects.taskmap, scheduler_state.time)

        return []
