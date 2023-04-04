################################################################################
# Cython implementations (Declarations are in device.pxd)
################################################################################

from parla.common.globals import _Locals as Locals
from parla.common.globals import cupy, CUPY_ENABLED
from parla.common.globals import DeviceType as PyDeviceType

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Union, List, Iterable, Dict, Tuple
from collections import defaultdict
import os 
from enum import IntEnum

cdef class CyDevice:
    """
    A bridge between pure Python and C++ device objects.
    """
    cdef Device* get_cpp_device(self):
        return self._cpp_device

    def __dealloc__(self):
        del self._cpp_device

    cpdef int get_global_id(self):
        return self._cpp_device.get_global_id()

    cpdef long long int query_resource(self, int resource_type):
        return self._cpp_device.query_resource(<Resource> resource_type)

    cpdef long long int query_reserved_resource(self, int resource_type):
        return self._cpp_device.query_reserved_resource(<Resource> resource_type)

    cpdef long long int query_mapped_resource(self, int resource_type):
        return self._cpp_device.query_mapped_resource(<Resource> resource_type)


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
        self._device_id = dev_id

        self._device = self 
        self._global_id = None 

    def __dealloc__(self):
        del self._cy_device

    def __enter__(self):
        return self 

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        #print(f"Exited device, {self.get_name()}, context", flush=True)

    @property
    def id(self):
        return self._dev_id 
    
    @id.setter
    def id(self, new_id):
        self._dev_id = new_id

    @property
    def global_id(self):
        return self._global_id
    
    @global_id.setter
    def global_id(self, new_id):
        self._global_id = new_id

    def __getitem__(self, param):
        if isinstance(param, Dict):
            memory_sz = 0 if "memory" not in param else param["memory"]
            num_vcus = 0 if "vcus" not in param else param["vcus"]
            return (self, DeviceResource(memory_sz, num_vcus))
        raise TypeError("[PyDevice] Parameter should be a dictionary specifying resource",
              " requirements.")

    def get_global_id(self):
        return self._cy_device.get_global_id()

    def get_name(self):
        return self._device_name

    def get_cy_device(self):
        return self._cy_device

    def query_resource(self, res_type):
        return self._cy_device.query_resource(res_type)

    def query_reserved_resource(self, res_type):
        return self._cy_device.query_reserved_resource(res_type)

    def query_mapped_resource(self, res_type):
        return self._cy_device.query_mapped_resource(res_type)

    @property
    def device(self):
        """
        Returns the external library device object if it exists (e.g. cupy for GPU devices).
        Otherwise, return the Parla device object (self).
        """
        return self._device

    @property
    def architecture(self):
        """
        Returns the architecture (type) of the device.
        """
        return self._dev_type

    def get_type(self):
        """
        Returns the architecture (type) of the device.
        """
        return self._dev_type

    def __repr__(self):
        return self._device_name

    def __hash__(self):
        #NOTE: DEVICE NAMES MUST BE UNIQUE INSIDE A SCHEDULER INSTANCE
        return hash(self._device_name)

    def __eq__(self, other):
        if isinstance(other, int):
            return self._dev_type == other
        elif isinstance(other, PyDevice):
            return self._device_name == other._device_name
        else:
            return False

    def __str__(self):
        return repr(self)

    @property
    def device_id(self):
        return self._device_id

    @property
    def id(self):
        return self._device_id


"""
Device instances in Python manage resource status.
TODO(hc): the device configuration will be packed in a data class soon.
"""

class PyCUDADevice(PyDevice):
    """
    An inherited class from `PyDevice` for a device object specialized to CUDA.
    """
    def __init__(self, dev_id: int, mem_sz: long, num_vcus: long):
        super().__init__(DeviceType.CUDA, "CUDA", dev_id)
        #TODO(wlr): If we ever support VECs, we might need to move this device initialization
        self._cy_device = CyCUDADevice(dev_id, mem_sz, num_vcus, self)

    @property
    def device(self):
        if CUPY_ENABLED:
            self._device = cupy.cuda.Device(self.device_id)
        return self._device


class PyCPUDevice(PyDevice):
    """
    An inherited class from `PyDevice` for a device object specialized to CPU.
    """
    def __init__(self, dev_id: int, mem_sz: long, num_vcus: long):
        super().__init__(DeviceType.CPU, "CPU", dev_id)
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
            error_msg = f"{self._name} does not have device({index})."
            error_msg += f" Please specify existing devices."
            raise ValueError(error_msg)

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
        if isinstance(o, int):
            return self.id == o
        elif isinstance(o, type(self)):
            return ( (self.id == o.id) and (self._name == o.name) )
        else:
            return False

    def __hash__(self):
        return self._id

    def __repr__(self):
        return type(self).__name__

    def __mul__(self, num_archs: int):
        arch_ps = [self for i in range(0, num_archs)]
        return tuple(arch_ps)

    def __len__(self):
        return len(self._devices)

 
class PyCUDAArchitecture(PyArchitecture):
    def __init__(self):
        super().__init__("CUDAArch", DeviceType.CUDA)

    def add_device(self, device):
        assert isinstance(device, PyDevice)
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


class Stream:
    def __init__(self, device=None, stream=None, non_blocking=True):
        self._device = device
        self._device_id = device.device.id
        self._stream = stream

    def __repr__(self):
        return f"Stream({self._device})"

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        #print("Entering Stream: ", self, flush=True)
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        #print("Exiting Stream: ", self, flush=True)
        pass

    @property
    def device(self):
        return self._device

    @property
    def stream(self):
        return self._stream

    def synchronize(self):
        print("Synchronizing stream", flush=True)

    def create_event(self):
        return None

    def wait_event(self):
        pass

class CupyStream(Stream):

    def __init__(self, device=None, stream=None, non_blocking=True):
        """
        Initialize a Parla Stream object.
        Assumes device and stream are cupy objects.
        """ 

        if device is None and stream is not None:
            raise ValueError("Device must be specified if stream is specified.")

        if device is None:
            self._device = cupy.cuda.Device()
            self._device_id = self._device.id
        else:
            self._device = device
            self._device_id = device.device.id

        with cupy.cuda.Device(self._device_id) as d:
            if stream is None:
                self._stream = cupy.cuda.Stream(non_blocking=non_blocking)
            else:
                self._stream = stream

    def __repr__(self):
        return f"Stream({self._device}, {self._stream})"

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        #print("Entering Stream: ", self, Locals.task, self._device_id, flush=True)

        #Set the device to the stream's device.
        self.active_device = cupy.cuda.Device(self._device_id)

        self.active_device.__enter__()
        #self._device.__enter__()

        
        #Set the stream to the current stream.
        self._stream.__enter__()

        Locals.push_stream(self)

        return self 

    def __exit__(self, exc_type, exc_value, traceback):

        ret_stream = False
        ret_device = False

        #Restore the stream to the previous stream.
        ret_stream = self._stream.__exit__(exc_type, exc_value, traceback)

        #Restore the device to the previous device.
        ret_device = self.active_device.__exit__(exc_type, exc_value, traceback)
            
        Locals.pop_stream()
        return ret_stream and ret_device

    @property
    def device(self):
        return self._device

    @property
    def stream(self):
        return self._stream

    def synchronize(self):
        print("Synchronizing stream", flush=True)
        self._stream.synchronize()

    def create_event(self):
        active_device = cupy.cuda.Device(self._device_id)
        with active_device:
            new_event = cupy.cuda.Event(block=True, disable_timing=True, interprocess=False)
        return new_event

    def wait_event(self, event):
        self._stream.wait_event(event)

    #TODO(wlr): What is the performance impact of this?
    def __getatrr__(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        return getattr(self._stream, name)
