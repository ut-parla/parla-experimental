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

from typing import List, Dict, Set, Tuple, Optional, Callable, Sequence
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict

from .scheduler import SchedulerArchitecture, SystemState, SchedulerOptions

from rich import print

StatesToResources: Dict[TaskState, list[ResourceType]] = {}
StatesToResources[TaskState.MAPPED] = [
    ResourceType.VCU,
    ResourceType.MEMORY,
    ResourceType.COPY,
]
StatesToResources[TaskState.LAUNCHED] = [ResourceType.VCU, ResourceType.COPY]
StatesToResources[TaskState.RESERVED] = [ResourceType.MEMORY]
StatesToResources[TaskState.COMPLETED] = []
AllResources = [ResourceType.VCU, ResourceType.MEMORY, ResourceType.COPY]


def map_task(task: SimulatedTask, scheduler_state: SystemState) -> Optional[Device]:
    objects = scheduler_state.objects
    assert objects is not None

    resource_pool = scheduler_state.resource_pool
    assert resource_pool is not None

    current_time = scheduler_state.time
    assert current_time is not None

    # Check if task is mappable
    if check_status := task.check_status(
        TaskStatus.MAPPABLE, objects.taskmap, current_time
    ):
        task.assigned_devices = (Device(Architecture.GPU, np.random.randint(0, 4)),)
        devices = task.assigned_devices
        # print(f"Task {task.name} assigned to device {devices}")
        assert devices is not None

        if devices is None:
            raise ValueError(
                f"Task {task.name} has no assigned devices. Minimal scheduler requires that all tasks have an assigned device at spawn."
            )
        device = devices[0]
        task.set_resources(device)

        # Update MAPPED resources (for policy and metadata)
        resource_pool.add_resources(
            devices=devices,
            state=TaskState.MAPPED,
            types=AllResources,
            resources=task.resources,
        )

        return device
    return None


def reserve_task(task: SimulatedTask, scheduler_state: SystemState) -> bool:
    objects = scheduler_state.objects
    assert objects is not None

    resource_pool = scheduler_state.resource_pool
    assert resource_pool is not None

    current_time = scheduler_state.time
    assert current_time is not None

    if check_reservable := task.check_status(
        TaskStatus.RESERVABLE, objects.taskmap, current_time
    ):
        devices = task.assigned_devices
        assert devices is not None

        resources_to_update = StatesToResources[TaskState.RESERVED]

        if can_fit := resource_pool.check_resources(
            devices=devices,
            state=TaskState.RESERVED,
            types=resources_to_update,
            resources=task.resources,
        ):
            resource_pool.add_resources(
                devices=devices,
                state=TaskState.RESERVED,
                types=resources_to_update,
                resources=task.resources,
            )
            return True
    return False


def launch_task(task: SimulatedTask, scheduler_state: SystemState) -> bool:
    objects = scheduler_state.objects
    assert objects is not None
    resource_pool = scheduler_state.resource_pool
    assert resource_pool is not None
    current_time = scheduler_state.time
    assert current_time is not None

    # Process LAUNCHABLE state
    if check_launchable_status := task.check_status(
        TaskStatus.LAUNCHABLE, objects.taskmap, current_time
    ):
        assert task.assigned_devices is not None
        resources_to_update = StatesToResources[TaskState.LAUNCHED]

        if check_launchable_resources := resource_pool.check_resources(
            devices=task.assigned_devices,
            state=TaskState.RESERVED,
            types=resources_to_update,
            resources=task.resources,
        ):
            # Update resource pool for RESERVED resources (for correctness)
            resource_pool.add_resources(
                devices=task.assigned_devices,
                state=TaskState.RESERVED,
                types=resources_to_update,
                resources=task.resources,
            )
            # Update resource pool for LAUNCHED resources (for metadata) (Tracks resources that are currently use)
            resource_pool.add_resources(
                devices=task.assigned_devices,
                state=TaskState.LAUNCHED,
                types=AllResources,
                resources=task.resources,
            )
            task.set_duration(task.assigned_devices, scheduler_state)
            return True
    return False


def complete_task(task: SimulatedTask, scheduler_state: SystemState) -> bool:
    resource_pool = scheduler_state.resource_pool
    assert resource_pool is not None

    devices = task.assigned_devices
    assert devices is not None

    # Free resources from all pools
    resource_pool.remove_resources(
        devices=devices,
        state=TaskState.MAPPED,
        types=AllResources,
        resources=task.resources,
    )
    resource_pool.remove_resources(
        devices=devices,
        state=TaskState.RESERVED,
        types=AllResources,
        resources=task.resources,
    )

    resource_pool.remove_resources(
        devices=devices,
        state=TaskState.LAUNCHED,
        types=AllResources,
        resources=task.resources,
    )

    return True


@SchedulerOptions.register_scheduler("parla")
@dataclass(slots=True)
class ParlaArchitecture(SchedulerArchitecture):
    topology: InitVar[SimulatedTopology]

    spawned_tasks: TaskQueue = TaskQueue()

    # Mapping Phase
    mappable_tasks: TaskQueue = TaskQueue()
    # Reserving Phase
    reservable_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)
    # Launching Phase
    launchable_tasks: Dict[Device, Dict[TaskType, TaskQueue]] = field(
        default_factory=dict
    )
    launched_tasks: Dict[Device, TaskQueue] = field(default_factory=dict)

    success_count: int = 0
    active_scheduler: int = 0

    def __post_init__(self, topology: SimulatedTopology):
        assert topology is not None

        for device in topology.devices:
            self.reservable_tasks[device.name] = TaskQueue()

            self.launchable_tasks[device.name] = dict()
            self.launchable_tasks[device.name][TaskType.DATA] = TaskQueue()
            self.launchable_tasks[device.name][TaskType.COMPUTE] = TaskQueue()

            self.launched_tasks[device.name] = TaskQueue()

    def initialize(
        self, tasks: List[TaskID], scheduler_state: SystemState
    ) -> Sequence[EventPair]:
        objects = scheduler_state.objects

        task_objects = [objects.get_task(task) for task in tasks]

        # Initialize the set of visible tasks
        self.add_initial_tasks(task_objects, scheduler_state)

        # Initialize the event queue
        next_event = Mapper()
        next_time = Time(0)
        self.active_scheduler += 1
        return [(next_time, next_event)]

    def add_initial_tasks(
        self, tasks: List[SimulatedTask], scheduler_state: SystemState
    ):
        """
        Append an initial task who does not have any dependency to
        a spawned task queue.
        """
        for task in tasks:
            self.spawned_tasks.put(task)

    def mapper(
        self, scheduler_state: SystemState, event: Mapper
    ) -> Sequence[EventPair]:
        # print("Mapping tasks...")
        self.success_count = 0
        next_tasks = TaskIterator(self.spawned_tasks)

        current_time = scheduler_state.time
        objects = scheduler_state.objects

        for priority, taskid in next_tasks:
            # print(f"Processing task {taskid}")

            task = objects.get_task(taskid)
            assert task is not None

            if device := map_task(task, scheduler_state):
                self.reservable_tasks[device].put_id(task_id=taskid, priority=priority)
                task.notify_state(TaskState.MAPPED, objects.taskmap, current_time)
                next_tasks.success()
                self.success_count += 1
            else:
                next_tasks.fail()
                continue

        reserver_pair = (current_time, Reserver())
        return [reserver_pair]

    def reserver(
        self, scheduler_state: SystemState, event: Reserver
    ) -> Sequence[EventPair]:
        objects = scheduler_state.objects
        current_time = scheduler_state.time

        next_tasks = MultiTaskIterator(self.reservable_tasks)
        for priority, taskid in next_tasks:
            task = objects.get_task(taskid)
            assert task is not None

            if reserve_success := reserve_task(task, scheduler_state):
                devices = task.assigned_devices
                assert devices is not None
                device = devices[0]
                self.launchable_tasks[device][task.type].put_id(
                    task_id=taskid, priority=priority
                )
                task.notify_state(TaskState.RESERVED, objects.taskmap, current_time)
                next_tasks.success()
                self.success_count += 1
            else:
                next_tasks.fail()
                continue

        launcher_pair = (current_time, Launcher())
        return [launcher_pair]

    def launcher(
        self, scheduler_state: SystemState, event: Launcher
    ) -> Sequence[EventPair]:
        # print("Launching tasks...")

        objects = scheduler_state.objects
        current_time = scheduler_state.time

        next_events: Sequence[EventPair] = []

        # print(f"Remaining tasks: {remaining_tasks}")

        next_tasks = MultiTaskIterator(self.launchable_tasks)
        for priority, taskid in next_tasks:
            task = objects.get_task(taskid)
            assert task is not None

            # Process LAUNCHABLE state
            if launch_success := launch_task(task, scheduler_state):
                task.notify_state(TaskState.LAUNCHED, objects.taskmap, current_time)
                completion_time = current_time + task.duration

                device = task.assigned_devices[0]  # type: ignore
                self.launched_tasks[device].put_id(taskid, completion_time)

                # Create completion event
                completion_event = TaskCompleted(task=taskid)
                next_events.append((completion_time, completion_event))
                next_tasks.success()
                self.success_count += 1
            else:
                next_tasks.fail()
                continue

        self.active_scheduler -= 1

        if remaining_tasks := length(self.launchable_tasks) and self.success_count:
            mapping_pair = (current_time + 100, Mapper())
            next_events.append(mapping_pair)
            self.active_scheduler += 1

        return next_events

    def _verify_correct_task_completed(
        self, task: SimulatedTask, scheduler_state: SystemState
    ):
        taskid = task.name
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
            self.completed_tasks.append(taskid)
        else:
            raise ValueError(
                f"Invalid state: Launch queue for device {device} is empty."
            )

    def complete_task(
        self, scheduler_state: SystemState, event: TaskCompleted
    ) -> Sequence[EventPair]:
        # print(f"Completing task: {event.task}...")
        objects = scheduler_state.objects
        task = objects.get_task(event.task)
        current_time = scheduler_state.time
        next_events: Sequence[EventPair] = []

        # print(task)
        # print(scheduler_state.resource_pool[Device(Architecture.GPU, 1)])
        # print(self)

        self._verify_correct_task_completed(task, scheduler_state)
        complete_task(task, scheduler_state)

        # Update status of dependencies
        task.notify_state(TaskState.COMPLETED, objects.taskmap, scheduler_state.time)
        # print(scheduler_state.resource_pool[Device(Architecture.GPU, 1)])

        self.success_count += 1
        if self.active_scheduler == 0:
            mapping_pair = (current_time + 100, Mapper())
            next_events.append(mapping_pair)
            self.active_scheduler += 1

        return next_events
