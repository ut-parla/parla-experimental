from parla.common.global_enums import DeviceType 

from dataclasses import dataclass

@dataclass
class DeviceConfig:
    """ Class for keeping track of device resources. """

    # Device name.
    dev_name: str
    # Device type (e.g., CPU or CUDA).
    dev_type: DeviceType
    # Id 0 is assigned to the first device of
    # each device type.
    dev_id: int
    num_vcus: int
    mem_sz: int
