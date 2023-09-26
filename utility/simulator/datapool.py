from __future__ import annotations
from ..types import DataID 
from .data import *
from .device import *
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Union, Self 

@dataclass(slots=True)
class DeviceDataPool:
    datamap: Dict[DataID, SimulatedData] = field(default_factory=dict)
    datalist: List[SimulatedData] = field(default_factory=list)
    memory: int = 0

    def __post_init__(self, devices: List[SimulatedDevice]):
        self.pool = {}

    def add_data(self, data: SimulatedData):
        self.datalist.append(data)
    
    def remove_data(self, data: SimulatedData):
        self.datalist.remove(data)

    

    



        