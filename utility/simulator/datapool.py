from __future__ import annotations
from ..types import DataID 
from .data import *
from .device import *
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Union, Self 

@dataclass(slots=True)
class DataNode:
    data: SimulatedData
    next: DataNode = None
    prev: DataNode = None
    
@dataclass(slots=True)
class DataNodeList:
    head: DataNode = None
    tail: DataNode = None
    size: int = 0
    map: Dict[DataID, DataNode] = field(default_factory=dict)

    def __post_init__(self):
        self.head = DataNode(None)
        self.tail = DataNode(None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def append(self, data: SimulatedData):
        node = DataNode(data)
        self.map[data.name] = node

        node.next = self.tail
        node.prev = self.tail.prev
        self.tail.prev.next = node
        self.tail.prev = node
        self.size += 1

    def remove(self, data: SimulatedData):
        node = self.map[data.name]
        del self.map[data.name]
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def __iter__(self):
        node = self.head.next
        while node is not self.tail:
            yield node.data
            node = node.next

    def __len__(self):
        return self.size

    def __str__(self):
        return f"DataNodeList({self.size})"

    def __repr__(self):
        return self.__str__()

@dataclass(slots=True)
class DeviceDataPool:
    datalist: DataNodeList = field(default_factory=DataNodeList)

    datamap: Dict[DataID, SimulatedData] = field(default_factory=dict)
    datalist: List[SimulatedData] = field(default_factory=list)
    memory: int = 0

    def add_data(self, data: SimulatedData):

        self.datalist.append(data)
    
    def remove_data(self, data: SimulatedData):
        self.datalist.remove(data)



    

    



        