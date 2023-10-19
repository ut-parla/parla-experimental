from __future__ import annotations
from ..types import DataID
from .data import *
from .device import *
from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Set, Tuple, Union, Self


@dataclass(slots=True)
class DataNode:
    data: Optional[DataInfo] = None
    next: Optional[DataNode] = None
    prev: Optional[DataNode] = None


@dataclass(slots=True)
class DataNodeList:
    head: DataNode = field(default_factory=DataNode)
    tail: DataNode = field(default_factory=DataNode)
    size: int = 0
    map: Dict[DataID, DataNode] = field(default_factory=dict)

    def __post_init__(self):
        self.head = DataNode()
        self.tail = DataNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def append(self, data: DataInfo):
        node = DataNode(data)
        self.map[data.id] = node

        node.next = self.tail
        node.prev = self.tail.prev
        self.tail.prev.next = node
        self.tail.prev = node
        self.size += 1

    def remove(self, data: DataInfo):
        node = self.map[data.id]
        del self.map[data.id]
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def __iter__(self):
        node = self.head.next
        assert node is not None
        assert node.data is not None

        while node.data is not None:
            yield node.data
            node = node.next
            assert node is not None

    def __len__(self):
        return self.size

    def __str__(self):
        return f"DataNodeList({self.size})"

    def __repr__(self):
        return self.__str__()


@dataclass(slots=True)
class EvictionPool:
    evictable_size: int = 0

    def add(self, data: SimulatedData):
        raise NotImplementedError

    def remove(self, data: SimulatedData):
        raise NotImplementedError

    def peek(self) -> DataID:
        raise NotImplementedError

    def get(self) -> DataID:
        raise NotImplementedError


@dataclass(slots=True)
class LRUEvictionPool(EvictionPool):
    datalist: DataNodeList = field(default_factory=DataNodeList)

    def add(self, data: SimulatedData):
        self.datalist.append(data.info)
        self.evictable_size += data.size

    def remove(self, data: SimulatedData):
        self.datalist.remove(data.info)
        self.evictable_size -= data.size
        assert self.evictable_size >= 0

    def peek(self) -> DataID:
        data = self.datalist.head.next.data
        assert data is not None
        return data.id

    def get(self) -> DataID:
        data = self.datalist.head.next.data
        assert data is not None
        self.datalist.remove(data)
        self.evictable_size -= data.size
        assert self.evictable_size >= 0
        return data.id


@dataclass(slots=True)
class DeviceDataPool:
    states2data: Dict[DataState, Set[DataID]] = field(default_factory=dict)
    evictable: EvictionPool = field(default_factory=LRUEvictionPool)

    def __post_init__(self):
        for state in [DataState.PLANNED, DataState.MOVING, DataState.VALID]:
            self.states2data[state] = set()

    def add_data(self, data: SimulatedData, state: DataState):
        if state not in self.states2data:
            self.states2data[state] = set()

        if state == DataState.EVICTABLE:
            self.evictable.add(data)
        else:
            self.states2data[state].add(data.name)

    def remove_data(self, data: SimulatedData, state: DataState):
        if state == DataState.EVICTABLE:
            self.evictable.remove(data)
        else:
            self.states2data[state].remove(data.name)

    def get_data_in_state(self, state: DataState) -> Sequence[DataID]:
        if state == DataState.EVICTABLE:
            return [list(d.info for d in self.evictable.datalist)]  # type: ignore
        else:
            datalist = self.states2data[state]
        return list(datalist)

    def get_evictable_size(self) -> int:
        return self.evictable.evictable_size

    def peek_evictable(self) -> DataID:
        return self.evictable.peek()

    def get_evictable(self) -> DataID:
        return self.evictable.get()

    def add_evictable(self, data: SimulatedData):
        self.evictable.add(data)

    def remove_evictable(self, data: SimulatedData):
        self.evictable.remove(data)


@dataclass(slots=True)
class DataPool:
    devices: InitVar[Sequence[SimulatedDevice]]
    devices2pools: Dict[Device, DeviceDataPool] = field(default_factory=dict)

    def __post_init__(self, devices: Sequence[SimulatedDevice]):
        for device in devices:
            self.devices2pools[device.name] = DeviceDataPool()

    def add_data(
        self, device: Device, data: SimulatedData, state: DataState, inital=False
    ):
        self.devices2pools[device].add_data(data, state)

    def remove_data(self, device: Device, data: SimulatedData, state: DataState):
        self.devices2pools[device].remove_data(data, state)
