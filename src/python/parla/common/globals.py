from enum import IntEnum
import threading
import os
try:
    import cupy
    CUPY_ENABLED = (os.getenv("PARLA_ENABLE_CUPY", "1") == "1")
except ImportError:
    cupy = None
    CUPY_ENABLED = False


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
    INVALID = -2
    ANY = -1
    CPU = 0
    CUDA = 1


class AccessMode(IntEnum):
    """
    This class declares PArray access modes that are used
    in @spawn.
    """
    IN = 0
    OUT = 1
    INOUT = 2


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
        return self._stack[-1]


class Locals(threading.local):

    def __init__(self):
        super(Locals, self).__init__()
        self._task_stack = LocalStack()
        self._context_stack = LocalStack()
        self._stream_stack = LocalStack()
        self._scheduler_stack = LocalStack()
        self._index = 0

    def push_task(self, task):
        self._task_stack.push(task)

    def pop_task(self):
        return self._task_stack.pop()

    @property
    def task(self):
        return self._task_stack.current 

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
    def devices(self):
        return self._context_stack.current.devices

    def push_scheduler(self, scheduler):
        self._scheduler_stack.push(scheduler)

    def pop_scheduler(self):
        return self._scheduler_stack.pop()

    @property
    def scheduler(self):
        return self._scheduler_stack.current

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value


_Locals = Locals()


def get_locals():
    return _Locals


def get_current_devices():
    return _Locals.devices


def get_active_device():
    return _Locals.active_device


def get_current_stream():
    return _Locals.stream


def get_current_context():
    return _Locals.context


def get_scheduler():
    return _Locals.scheduler


def get_device_manager():
    return get_scheduler().device_manager


def get_stream_pool():
    return get_device_manager().stream_pool


def has_environment():
    return True if _Locals.context else False
