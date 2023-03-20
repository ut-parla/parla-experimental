from enum import IntEnum


class DeviceType(IntEnum):
    """
    This class declares device types.
    These types are declared for the general purpose
    and can be used by multiple places.
    For example,
     1) Task mapping phase can check compatibility
        between tasks and devices.
     2) Tasks hold resource requirements from mapped
        devices through a device set data class.
        Device set data class instances hold resource 
        requirement for each device. This device class
        can be distinguished through these types.
     3) Device registration phase can utilize these types.
    """
    ANY = -1
    INVALID = 2
    CUDA = 1
    CPU = 0


