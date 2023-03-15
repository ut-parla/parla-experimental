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

    @property
    def device(self):
        """
        Returns the external library device object if it exists (e.g. cupy for GPU devices).
        Otherwise, return the Parla device object (self).
        """
        return self._device

    def get_type(self):
        return self._dev_type

    def __repr__(self):
        return self._device_name

    def __hash__(self):
        #NOTE: DEVICE NAMES MUST BE UNIQUE IN A SCHEDULER INSTANCE
        return hash(self._device_name)


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
        #TODO(wlr): If we ever support VECs, we might need to move this
        self._device = cupy.cuda.Device(dev_id)
        self._cy_device = CyCUDADevice(dev_id, mem_sz, num_vcus, self)


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


import threading 

class StreamPool:

    def __init__(self, device_list, per_device=4):
        self._device_list = device_list
        self._per_device = per_device
        self._pool = {}
        print("Device List: ", device_list)
        for device in self._device_list:
            self._pool[device] = []
            for i in range(self._per_device):
                self._pool[device].append(Stream(device=device.device))

    def get_stream(self, device):
        if len(self._pool[device]) == 0:
            #Create a new stream if the pool is empty.
            return Stream(device=device)
        return self._pool[device].pop()

    def return_stream(self, stream):
        self._pool[stream.device].append(stream)

    def __summarize__(self):
        summary  = ""
        for device in self._device_list:
            summary += f"({device} : {len(self._pool[device])})"

        return summary

    def __repr__(self):
        return f"StreamPool({self.__summarize__()})"

class Stream:

    def __init__(self, device=None, stream=None, non_blocking=True):
        """
        Initialize a Parla Stream object.
        Assumes device and stream are cupy objects.
        """ 

        if device is None and stream is not None:
            raise ValueError("Device must be specified if stream is specified.")

        if device is None:
            self._device = cupy.cuda.Device()
        else:
            self._device = device

        if stream is None:
            self._stream = cupy.cuda.Stream(non_blocking=non_blocking)
        else:
            self._stream = stream

    def __repr__(self):
        return f"Stream({self._device}, {self._stream})"

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        #Set the device to the stream's device.
        self._device.__enter__()

        #Set the stream to the current stream.
        self._stream.__enter__()

        return self 

    def __exit__(self, exc_type, exc_value, traceback):

        ret_stream = False
        ret_device = False

        try:
            #Restore the stream to the previous stream.
            ret_stream = self._stream.__exit__(exc_type, exc_value, traceback)

            #Restore the device to the previous device.
            ret_device = self._device.__exit__(exc_type, exc_value, traceback)
        finally:
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

    #TODO(wlr): What is the performance impact of this?
    def __getatrr__(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        return getattr(self._stream, name)


class LocalStack(threading.local):

    def __init__(self):
        super(LocalStack, self).__init__()
        self._stack = []

    def __repr__(self):
        return str(self._stack)

    def __str__(self):
        return self.__repr__()

    def push(self, context):
        self._stack.append(context)

    def pop(self):
        return self._stack.pop()

    @property
    def current(self):
        if len(self._stack) == 0:
            return None
        return self._context_stack[-1]

class Locals(threading.local):

    def __init__(self):
        super(Locals, self).__init__()
        self._context_stack = LocalStack()
        self._stream_stack = LocalStack()

    def add_stream_pool(self, stream_pool):
        self._stream_pool = stream_pool

    @property
    def stream_pool(self):
        if not hasattr(self, "_stream_pool"):
            raise ValueError("Stream pool not initialized.")
        return self._stream_pool

    def push_context(self, context):
        self._context_stack.push(context)

    def pop_context(self):
        return self._context_stack.pop()

    @property
    def context(self):
        return self._context_stack.current

    def push_stream(self, stream):
        self._stream_stack.push(stream)
    
    def pop_stream(self):
        return self._stream_stack.pop()

    @property
    def stream(self):
        return self._stream_stack.current

    @property
    def active_device(self):
        return self._stream_stack.current.device

    @property
    def current_devices(self):
        return self._context_stack.current.devices


def create_device_env(device):
    if isinstance(device, PyCPUDevice):
        return TaskEnvironment([device]), DeviceType.CPU
    elif isinstance(device, PyCUDADevice):
        return GPUEnvironment([device]), DeviceType.CUDA

def create_env(sources):
    targets = []

    for env in sources:
        if isinstance(env, PyDevice):
            device = env
            new_env, dev_type = create_device_env(env)
            targets.append(new_env)

    if len(targets) == 1:
        return targets[0]
    else:
        return TaskEnvironment(targets)

_Locals = Locals()

class TaskEnvironment:

    def __init__(self, environment_list, blocking=False):

        self.device_dict = defaultdict(list)
        self.env_list = []
        self.is_terminal = False
        self.blocking = blocking

        for env in environment_list:
            for dev in env.device_dict:
                self.device_dict[dev] += env.device_dict[dev]
            self.env_list.append(env)

    def __repr__(self):
        return f"TaskEnvironment({self.env_list})"

    @property
    def streams(self):
        if self.is_terminal:
            return self.stream_list
        else:
            return None

    def get_devices(self, arch):
        return self.device_dict[arch]

    def get_all_devices(self):
        return list(self.device_dict.values())

    def get_cupy_devices(self):
        return [dev.device for dev in self.get_devices(DeviceType.CUDA)]

    def synchronize(self):
        print(f"Sychronizing {self}..", flush=True)
        if self.is_terminal:
            for stream in self.stream_list:
                stream.synchronize()
        else:
            for env in self.env_list:
                env.synchronize()

    def __enter__(self):
        print("Entering environment:", self.env_list, flush=True)

        if len(self.env_list) == 0:
            raise RuntimeError("[TaskEnvironment] No environment or device is available.")

        _Locals.push_context(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting environment", self.env_list, flush=True)
        ret = True

        _Locals.pop_context(self)
        
        return ret 

    def __getitem__(self, index):

        if isinstance(index, int):
            return self.env_list[index]
        
        return create_env(self.env_list[index])


class GPUEnvironment(TaskEnvironment):

    def __init__(self, device, blocking=False):
        super(GPUEnvironment, self).__init__([], blocking=blocking)

        self.stream_list = []
        self.is_terminal = True

        self.device_dict[DeviceType.CUDA].append(device)

        self.device = device 
        stream = Locals._stream_pool.get_stream(device=device)
        self.stream_list.append(stream)

    def __repr__(self):
        return f"GPUEnvironment({self.env_list})"

    def __enter__(self):
        print("Entering GPU Environment", flush=True)
        if len(self.env_list) == 0:
            raise RuntimeError("[TaskEnvironment] No environment or device is available.")

        _Locals.push_context(self)

        stream = self.stream_list[0]

        ret_stream = stream.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting GPU Environment", flush=True)
        ret = True

        for stream in self.stream_list:
            Locals.pop_stream(stream)

        _Locals.pop_context(self)
        
        return ret 

    def __getitem__(self, index):

        if isinstance(index, int):
            return self.env_list[index]
        
        return create_env(self.env_list[index])


    def finalize(self):
        for stream in self.stream_list:
            stream.synchronize()
            Locals._stream_pool.return_stream(stream)