from typing import Type
from ..common.globals import cupy
from parla.common.globals import _Locals as Locals

"""
cdef class Stream:
    cdef object _stream
    cdef object _device
    
    def __init__(self, device = None , stream=None):
        self._stream = stream
        self._device = device

    def __repr__(self):
        return f"Stream({self._device})"

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        # print("Entering Stream: ", self, flush=True)
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # print("Exiting Stream: ", self, flush=True)
        pass

    @property
    def device(self):
        return self._device

    @property
    def stream(self):
        return self._stream

    def synchronize(self):
        pass

    def create_event(self):
        return None

    def wait_event(self):
        pass

    @property
    def ptr(self):
        return None


cdef class CupyStream(Stream):
    
    def __init__(self, device = None , stream = None):
        super().__init__(device=device, stream=stream)
"""
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
        # print("Entering Stream: ", self, flush=True)
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # print("Exiting Stream: ", self, flush=True)
        pass

    @property
    def device(self):
        return self._device

    @property
    def stream(self):
        return self._stream

    def synchronize(self):
        pass

    def create_event(self):
        return None

    def wait_event(self):
        pass

    @property
    def ptr(self):
        return None


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
        # Set the device to the stream's device.
        self.active_device = cupy.cuda.Device(self._device_id)

        self.active_device.__enter__()
        # self._device.__enter__()

        # Set the stream to the current stream.
        self._stream.__enter__()

        Locals.push_stream(self)

        return self 

    def __exit__(self, exc_type, exc_value, traceback):

        ret_stream = False
        ret_device = False

        # Restore the stream to the previous stream.
        ret_stream = self._stream.__exit__(exc_type, exc_value, traceback)

        # Restore the device to the previous device.
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
        # print("Synchronizing stream", flush=True)
        self._stream.synchronize()

    def create_event(self):
        active_device = cupy.cuda.Device(self._device_id)
        with active_device:
            new_event = cupy.cuda.Event(block=True, disable_timing=True, interprocess=False)
        return new_event

    def wait_event(self, event):
        self._stream.wait_event(event)

    @property
    def ptr(self):
        return self._stream.ptr

    # TODO(wlr): What is the performance impact of this?
    def __getatrr__(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        return getattr(self._stream, name)


cdef class CyStreamPool:

    def __cinit__(self):
        self._c_pool = new InnerStreamPool()

    def __dealloc__(self):
        del self._c_pool

    def __init__(self, device_list, per_device=8, cupy_flag=False):

        if cupy_flag:
            self.StreamClass = CupyStream 
        else:
            self.StreamClass = Stream

        self._device_list = device_list
        self._per_device = per_device
        self._pool = {}

        for device in self._device_list:
            self._pool[device] = []
            
            with device.device as d:
                for i in range(self._per_device):
                    self._pool[device].append(self.StreamClass(device=device))

    def get_stream(self, device):
        if len(self._pool[device]) == 0:
            # Create a new stream if the pool is empty.
            new_stream = self.StreamClass(device=device)
            return new_stream

        return self._pool[device].pop()

    def return_stream(self, stream):
        self._pool[stream.device].append(stream)

    def __summarize__(self):
        summary = ""
        for device in self._device_list:
            summary += f"({device} : {len(self._pool[device])})"

        return summary

    def __repr__(self):
        return f"StreamPool({self.__summarize__()})"


class StreamPool:

    def __init__(self, device_list, per_device=8, cupy_flag: bool = True):

        if cupy_flag:
            self.StreamClass = CupyStream 
        else:
            self.StreamClass = Stream

        self._device_list = device_list
        self._per_device = per_device
        self._pool = {}

        for device in self._device_list:
            self._pool[device] = []
            
            with device.device as d:
                for i in range(self._per_device):
                    self._pool[device].append(self.StreamClass(device=device))

    def get_stream(self, device):
        if len(self._pool[device]) == 0:
            # Create a new stream if the pool is empty.
            new_stream = self.StreamClass(device=device)
            return new_stream

        return self._pool[device].pop()

    def return_stream(self, stream):
        self._pool[stream.device].append(stream)

    def __summarize__(self):
        summary = ""
        for device in self._device_list:
            summary += f"({device} : {len(self._pool[device])})"

        return summary

    def __repr__(self):
        return f"StreamPool({self.__summarize__()})"
