from ..types import (
    Architecture,
    Device,
    TaskID,
    DataID,
    DataInfo,
    TaskState,
    TaskStatus,
    AccessType,
)
from typing import List, Dict, Set, Tuple, Optional, Sequence
from dataclasses import dataclass, field, InitVar
from collections import defaultdict as DefaultDict
from enum import IntEnum


class DataMovementFlags(IntEnum):
    """
    Status of data movement to a device
    """

    FIRST_MOVE = 0
    """ Data is being moved to the device for the first time """
    ALREADY_MOVING = 1
    """ Data is already being moved to the device """
    ALREADY_THERE = 2
    """ Data is already on the device due to a previous move """


class DataState(IntEnum):
    """
    Internal state of data on a device (not all states are used)
    Note: Not to be confused with TaskState
    """

    NONE = -1
    """ Data is not present on the device """
    PLANNED = 0
    """ Data is planned to be present on the device (not on the device yet, but requested by a task)"""
    MOVING = 1
    """ Data is in transit to the device """
    VALID = 2
    """ Data is valid on the device """
    EVICTABLE = 3
    """ Data is evictable from the device """
    STALE = 3
    """ Data is stale on the device """


class DataUses(IntEnum):
    """
    State representing how data is being used by a task
    """

    MAPPED = 1
    """ A mapped compute task is using the data """
    RESERVED = 2
    """ A reserved compute task is using the data """
    MOVING = 3
    """ A data task is currently moving the data to OR from the device """
    USED = 4
    """ A launched compute task is using the data """


TaskStateToUse = {}
TaskStateToUse[TaskState.MAPPED] = DataUses.MAPPED
TaskStateToUse[TaskState.RESERVED] = DataUses.RESERVED
TaskStateToUse[TaskState.LAUNCHED] = DataUses.USED

NonEvictableUses = [DataUses.RESERVED, DataUses.MOVING, DataUses.USED]


@dataclass(slots=True)
class DataUse:
    tasks: Dict[DataUses, Set[TaskID]] = field(default_factory=dict)
    counters: Dict[DataUses, int] = field(default_factory=dict)

    def __post_init__(self):
        for use in DataUses:
            self.tasks[use] = set()
            self.counters[use] = 0

    def is_evictable(self):
        for use in NonEvictableUses:
            if self.counters[use] > 0:
                return False
        return True

    def is_used(self, use: DataUses) -> bool:
        return self.counters[use] > 0

    def get_use_count(self, use: DataUses) -> int:
        return self.counters[use]

    def add_task(self, task: TaskID, use: DataUses):
        self.tasks[use].add(task)
        self.counters[use] += 1

    def remove_task(self, task: TaskID, use: DataUses):
        self.tasks[use].remove(task)
        self.counters[use] -= 1

    def __rich_repr__(self):
        yield "tasks", self.tasks


@dataclass(slots=True)
class DataStatus:
    devices: InitVar[Sequence[Device]]
    device2state: Dict[TaskState, Dict[Device, DataState]] = field(default_factory=dict)
    state2device: Dict[TaskState, Dict[DataState, Set[Device]]] = field(
        default_factory=dict
    )
    device2uses: Dict[Device, DataUse] = field(default_factory=dict)

    def __post_init__(self, devices: Sequence[Device]):
        for state in TaskState:
            self.device2state[state] = {}
            self.state2device[state] = {}

            for device in devices:
                self.device2state[state][device] = DataState.NONE

            for data_state in DataState:
                self.state2device[state][data_state] = set()

        for device in devices:
            self.device2uses[device] = DataUse()

    def set_state(
        self, device: Device, state: TaskState, data_state: DataState, initial=False
    ):
        if not initial:
            prior_state = self.device2state[state][device]
            self.state2device[state][prior_state].remove(device)

        self.device2state[state][device] = data_state
        self.state2device[state][data_state].add(device)

    def check_state(
        self, device: Device, state: TaskState, data_state: DataState
    ) -> bool:
        return self.device2state[state][device] == data_state

    def get_state(self, device: Device, state: TaskState) -> DataState:
        return self.device2state[state][device]

    def get_use(self, device: Device) -> DataUse:
        return self.device2uses[device]

    def get_devices(
        self, state: TaskState, data_states: Sequence[DataState]
    ) -> Sequence[Device]:
        devices = []
        for data_state in data_states:
            devices.extend(self.state2device[state][data_state])
        return devices

    def add_task(self, device: Device, task: TaskID, use: DataUses):
        self.device2uses[device].add_task(task, use)

    def remove_task(self, device: Device, task: TaskID, use: DataUses):
        self.device2uses[device].remove_task(task, use)

    def is_evictable(self, device: Device) -> bool:
        return self.device2uses[device].is_evictable()

    def is_used(self, device: Device, use: DataUses) -> bool:
        return self.device2uses[device].is_used(use)

    def get_use_count(self, device: Device, use: DataUses) -> int:
        return self.device2uses[device].get_use_count(use)

    def advance_state(
        self,
        device: Device,
        state: TaskState,
        new_data_state: DataState,
        force: bool = False,
    ) -> bool:
        if force:
            self.set_state(device, state, new_data_state)
            return True  # Change
        else:
            prior_data_state = self.get_state(device, state)
            if new_data_state > prior_data_state:
                self.set_state(device, state, new_data_state)
                return True  # Change
        return False  # No change

    def verify_write(self, device: Device, state: TaskState, check_use: bool = True):
        status = self.device2state[state]

        # Ensure no device is moving the data
        for device in status.keys():
            if status[device] == DataState.MOVING:
                raise RuntimeError(
                    f"Cannot write while a device that is moving that data. Status: {status}"
                )

        # Ensure no device is using the data if check_use is True
        if check_use:
            for device in status.keys():
                if self.is_used(device, DataUses.USED):
                    raise RuntimeError(
                        f"Cannot write while a device that is using that data. Status: {status}"
                    )

    def write(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        verify: bool = False,
        update: bool = False,
    ):
        if verify:
            # Assumes that this happens before the task is added to the device uses list
            self.verify_write(target_device, state)

        # Invalidate all other devices and check that the target device is valid
        status = self.device2state[state]
        for device in status.keys():
            if device == target_device:
                if update:
                    self.set_state(device, state, DataState.VALID)
                else:
                    if not self.check_state(device, state, DataState.VALID):
                        raise RuntimeError(
                            f"Task {task} cannot write to data that is not valid. Status: {status}"
                        )
            else:
                self.set_state(device, state, DataState.NONE)

    def read(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        update: bool = False,
    ):
        # Ensure that the target device is valid
        status = self.device2state[state]

        if update:
            self.set_state(target_device, state, DataState.VALID)
        else:
            if not self.check_state(target_device, state, DataState.VALID):
                raise RuntimeError(
                    f"Task {task} cannot read from data that is not valid. Status: {status}"
                )

    def start_use(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        operation: AccessType,
        update: bool = False,
    ):
        if operation == AccessType.READ:
            self.read(
                task=task, target_device=target_device, state=state, update=update
            )
        else:
            self.write(
                task=task, target_device=target_device, state=state, update=update
            )

        self.add_task(target_device, task, TaskStateToUse[state])

    def finish_use(self, task: TaskID, target_device: Device, state: TaskState):
        self.remove_task(target_device, task, TaskStateToUse[state])

    def start_move(
        self, task: TaskID, source_device: Device, target_device: Device
    ) -> DataMovementFlags:
        if not self.check_state(source_device, TaskState.LAUNCHED, DataState.VALID):
            raise RuntimeError(
                f"Task {task} cannot move data from a device that is not valid."
            )

        old_state = self.get_state(target_device, TaskState.LAUNCHED)

        if self.check_state(target_device, TaskState.LAUNCHED, DataState.VALID):
            movement_flag = DataMovementFlags.ALREADY_THERE
        elif self.check_state(target_device, TaskState.LAUNCHED, DataState.MOVING):
            movement_flag = DataMovementFlags.ALREADY_MOVING
        else:
            movement_flag = DataMovementFlags.FIRST_MOVE
            self.set_state(target_device, TaskState.LAUNCHED, DataState.MOVING)

        self.add_task(source_device, task, DataUses.MOVING)
        self.add_task(target_device, task, DataUses.MOVING)

        return movement_flag

    def finish_move(
        self, task: TaskID, source_device: Device, target_device: Device
    ) -> DataMovementFlags:
        if self.check_state(target_device, TaskState.LAUNCHED, DataState.VALID):
            movement_flag = DataMovementFlags.ALREADY_THERE
        elif self.check_state(target_device, TaskState.LAUNCHED, DataState.MOVING):
            self.set_state(target_device, TaskState.LAUNCHED, DataState.VALID)
            movement_flag = DataMovementFlags.FIRST_MOVE
        else:
            raise RuntimeError(
                f"Task {task} cannot finish moving data to a device that is not moving."
            )

        self.remove_task(source_device, task, DataUses.MOVING)
        self.remove_task(target_device, task, DataUses.MOVING)

        return movement_flag

    def __rich_repr__(self):
        yield "MAPPED", self.device2state[TaskState.MAPPED]
        yield "RESERVED", self.device2state[TaskState.RESERVED]
        yield "LAUNCHED", self.device2state[TaskState.LAUNCHED]


@dataclass(slots=True)
class SimulatedData:
    system_devices: InitVar[Sequence[Device]]
    info: DataInfo
    status: DataStatus = field(init=False)

    def __post_init__(self, system_devices: Sequence[Device]):
        self.status = DataStatus(system_devices)

        starting_devices = self.info.location
        assert starting_devices is not None

        if isinstance(starting_devices, Device):
            starting_devices = (starting_devices,)

        for device in starting_devices:
            for state in TaskState:
                self.status.set_state(device, state, DataState.VALID, initial=True)

    @property
    def name(self) -> DataID:
        return self.info.id

    @property
    def size(self) -> int:
        return self.info.size

    def get_state(self, device: Device, state: TaskState) -> DataState:
        return self.status.get_state(device, state)

    def start_use(
        self,
        task: TaskID,
        target_device: Device,
        state: TaskState,
        operation: AccessType,
    ):
        self.status.start_use(task, target_device, state, operation)

    def finish_use(self, task: TaskID, target_device: Device, state: TaskState):
        self.status.finish_use(task, target_device, state)

    def start_move(
        self, task: TaskID, source_device: Device, target_device: Device
    ) -> DataMovementFlags:
        return self.status.start_move(task, source_device, target_device)

    def finish_move(
        self, task: TaskID, source_device: Device, target_device: Device
    ) -> DataMovementFlags:
        return self.status.finish_move(task, source_device, target_device)

    def __str__(self):
        return f"Data({self.name}) | Status: {self.status}"

    def __repr__(self):
        return self.__str__()

    def __rich_repr__(self):
        yield "info", self.info
        yield "status", self.status
        yield "used_by", self.status.device2uses

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


SimulatedDataMap = Dict[DataID, SimulatedData]
