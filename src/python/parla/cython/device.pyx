################################################################################
# Cython implementations (Declarations are in device.pxd)
################################################################################

from parla.common.globals import DeviceType, cupy

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Union, List, Iterable, Dict, Tuple
from collections import defaultdict

cdef class CyDevice:
    """
    A bridge between pure Python and C++ device objects.
    """
    cdef Device* get_cpp_device(self):
        return self._cpp_device


cdef class CyCUDADevice(CyDevice):
    """
    An inherited class from `CyDevice` for a device object specialized to CUDA.
    """
    def __cinit__(self, int dev_id, long mem_sz, long num_vcus, py_device):
        # C++ device object.
        # This object is deallocated by the C++ device manager.
        self._cpp_device = new CUDADevice(dev_id, mem_sz, num_vcus, \
                                          <void *> py_device)

    def __init__(self, int dev_id, long mem_sz, long num_vcus, py_device):
        pass


cdef class CyCPUDevice(CyDevice):
    """
    An inherited class from `CyDevice` for a device object specialized to CPU.
    """
    def __cinit__(self, int dev_id, long mem_sz, long num_vcus, py_device):
        # C++ device object.
        # This object is deallocated by the C++ device manager.
        self._cpp_device = new CPUDevice(dev_id, mem_sz, num_vcus, \
                                         <void *> py_device)

    def __init__(self, int dev_id, long mem_sz, long num_vcus, py_device):
        pass


################################################################################
# Python
################################################################################


class DeviceResource:
    def __init__(self, memory_sz = 0, num_vcus = 0):
        # This class represents a device total resource size.
        # This can also be used to specify resource requirements
        # of a task for task mapping. 
        # 0 value implies that there is no constraint in a
        # resource. In the same sense, 0 value in a requirement
        # implies that it can be mapped to a device even though 
        # that device does not have enough resource.
        # TODO(hc): better design? map still has a problem that
        #           users should remember keys.
        self.memory_sz = memory_sz
        self.num_vcus = num_vcus


class PyDevice:
    """
    This class is to abstract a single device in Python and manages
    a device context as a task runs in Python.
    """
    def __init__(self, dev_type, dev_type_name, dev_id: int):
        self._dev_type = dev_type
        self._device_name = dev_type_name + ":" + str(dev_id)
        self._device = self 

    def __enter__(self):
        pass
        #print(f"Entered device, {self.get_name()}, context", flush=True)

    def __exit__(self):
        pass
        #print(f"Exited device, {self.get_name()}, context", flush=True)

    def __getitem__(self, param):
        if isinstance(param, Dict):
            memory_sz = 0 if "memory" not in param else param["memory"]
            num_vcus = 0 if "vcus" not in param else param["vcus"]
            return (self, DeviceResource(memory_sz, num_vcus))
        raise TypeError("[PyDevice] Parameter should be a dictionary specifying resource",
              " requirements.")

    def get_name(self):
        return self._device_name

    def get_cy_device(self):
        return self._cy_device

    def get_device(self):
        return self._device

    def get_type(self):
        return self._dev_type

    def __repr__(self):
        return self._device_name


"""
Device instances in Python manage resource status.
TODO(hc): the device configuration will be packed in a data class soon.
"""

class PyCUDADevice(PyDevice):
    """
    An inherited class from `PyDevice` for a device object specialized to CUDA.
    """
    def __init__(self, dev_id: int, mem_sz: long, num_vcus: long):
        super().__init__(DeviceType.CUDA, "CUDADev", dev_id)
        #TODO(wlr): If we ever support VECs, we might need to move this
        self._device = cupy.cuda.Device(dev_id)
        self._cy_device = CyCUDADevice(dev_id, mem_sz, num_vcus, self)


class PyCPUDevice(PyDevice):
    """
    An inherited class from `PyDevice` for a device object specialized to CPU.
    """
    def __init__(self, dev_id: int, mem_sz: long, num_vcus: long):
        super().__init__(DeviceType.CPU, "CPUDev", dev_id)
        self._cy_device = CyCPUDevice(dev_id, mem_sz, num_vcus, self)


class PyArchitecture(metaclass=ABCMeta):
    """
    This class is to abstract a single architecture and is utilized for
    two purposes.
    First, an architecture class holds and provides device instances.
    (through a device manager)
    Second, users can specify the architecture at task spawn's placement
    parameter.
    """

    def __init__(self, name, id):
        """
        Create a new Architecture with a name and the ID which the runtime
        will use to identify it.
        """
        self._name = name
        self._id = id
        self._devices = []

    def __call__(self, index, *args, **kwds):
        """
        Create a device with this architecture.
        The arguments can specify which physical device you are requesting,
        but the runtime may override you.

        >>> gpu(0)
        """
        try:
            return self._devices[index]
        except IndexError:
            # If a requested device does not exist,
            # ignore that placement.
            print(f"{self._name} does not have device({index}).", flush=True)
            print(f"Ignore this placement.", flush=True)
            return None

    def __getitem__(self, param):
        if isinstance(param, Dict):
            memory_sz = 0 if "memory" not in param else param["memory"]
            num_vcus = 0 if "vcus" not in param else param["vcus"]
            return (self, DeviceResource(memory_sz, num_vcus))
        raise TypeError("[PyArchitecture] Parameter should be a dictionary specifying resource",
              " requirements.")

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def devices(self):
        """
        :return: all `devices<Device>` with this architecture in the system.
        """
        return self._devices

    def __eq__(self, o: object) -> bool:
        return isinstance(o, type(self)) and \
               self.id == o.id and self._name == o.name

    def __hash__(self):
        return self._id

    def __repr__(self):
        return type(self).__name__

 
class PyCUDAArchitecture(PyArchitecture):
    def __init__(self):
        super().__init__("CUDAArch", DeviceType.CUDA)

    def add_device(self, device):
        assert isinstance(device, PyCUDADevice)
        self._devices.append(device)

 
class PyCPUArchitecture(PyArchitecture):
    def __init__(self):
        super().__init__("CPUArch", DeviceType.CPU)

    def add_device(self, device):
        assert isinstance(device, PyCPUDevice)
        self._devices.append(device)


# TODO(hc): use dataclass later.
class DeviceResourceRequirement:
    def __init__(self, device: PyDevice, res_req: DeviceResource):
        self.device = device
        self.res_req = res_req

    def __repr__(self):
        return "("+self.device.get_name()+", memory:"+str(self.res_req.memory_sz)+ \
               ", num_vcus:"+str(self.res_req.num_vcus)+")" 

PlacementSource = Union[PyArchitecture, PyDevice, Tuple[PyArchitecture, DeviceResource], \
                        Tuple[PyDevice, DeviceResource]]

def create_device_env(device):
    if isinstance(device, PyCPUDevice):
        return TaskEnvironment([device]), DeviceType.CPU
    elif isinstance(device, PyCUDADevice):
        return TaskEnvironment([device]), DeviceType.CUDA
    
class TaskEnvironment:

    def __init__(self, environment_list):

        self.device_dict = defaultdict(list)
        self.env_list = []

        for env in environment_list:
            if isinstance(env, PyDevice):
                dev = env
                env, dev_type = create_device_env(env)
                self.device_dict[dev_type].append(env)
                self.env_list.append(env) 
            elif isinstance(env, PyArchitecture):
                for dev in env.devices:
                    env, dev_type = create_device_env(dev)
                    self.device_dict[dev_type].append(env)
                    self.env_list.append(dev)
            elif isinstance(env, TaskEnvironment):
                for dev in env.device_dict:
                    self.device_dict[dev] += env.device_dict[dev]
                self.env_list.append(env)
            else:
                raise TypeError("[TaskEnvironment] Unsupported environment type.")

    def __enter__(self):

        if len(self.env_list) == 0:
            raise RuntimeError("[TaskEnvironment] No environment or device is available.")
        else:
            self._stack.append(self.env_list[0].__enter__())




    def __getitem__(self, index):

        if isinstance(index, int):
            return self.env_list[index]
        elif isinstance(index, PyArchitecture):
            return TaskEnvironment(self.device_dict[index])
        
        return TaskEnvironment(self.env_list[index])

