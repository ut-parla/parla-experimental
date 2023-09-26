from ..types import Architecture, Device, TaskID, TaskState, ResourceType
from dataclasses import dataclass, field
from .queue import PriorityQueue
from enum import IntEnum
from typing import List, Dict, Set, Tuple, Optional, Self
from fractions import Fraction
from decimal import Decimal
from collections import defaultdict as DefaultDict

Numeric = int | float | Fraction | Decimal


@dataclass(slots=True)
class ResourceSet:
    store: DefaultDict[ResourceType, Numeric] = field(
        default_factory=lambda: DefaultDict(int))

    def __post_init__(self, vcus: Numeric, memory: int, copy: int):
        self.store[ResourceType.VCUS] = Fraction(vcus)
        self.store[ResourceType.MEMORY] = memory
        self.store[ResourceType.COPY] = copy

    def __getitem__(self, key: ResourceType) -> Numeric:
        return self.store[key]

    def __setitem__(self, key: ResourceType, value: Numeric):
        self.store[key] = value

    def __add__(self, other: Self) -> Self:
        for key in other.store:
            if key in self.store:
                self.store[key] += other.store[key]

    def __sub__(self, other: Self) -> Self:
        for key in other.store:
            if key in self.store:
                self.store[key] -= other.store[key]

    def verify(self):
        for key in self.store:
            if self.store[key] < 0:
                raise ValueError(
                    f"ResourceSet {self} contains negative value for {key}")

    def __str__(self) -> str:
        return f"ResourceSet({self.store})"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: Self) -> bool:
        for key in other.store:
            if self.store[key] >= other.store[key]:
                return False
        return True

    def __eq__(self, other: Self) -> bool:
        for key in other.store:
            if self.store[key] != other.store[key]:
                return False
        return True


class DataPool:
    pass


@dataclass(slots=True)
class SimulatedDevice:
    name: Device
    tasks: Dict[TaskState, PriorityQueue[TaskID]]
    resources: ResourceSet = ResourceSet(1, 100, 2)
