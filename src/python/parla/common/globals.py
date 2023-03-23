from enum import IntEnum
import threading
import os

try:
    import cupy
    CUPY_ENABLED = (os.getenv("PARLA_ENABLE_CUPY", "1") == "1")
except ImportError:
    cupy = None
    CUPY_ENABLED = False


class SynchronizationType(IntEnum):
    """
    This class declares the type of event synchronization used for runahead scheduling.
    """

    # No runahaed scheduling is used.
    NONE = 0

    # Default events are recorded for each stream in the task.
    # The host waits for all event dependencies to be completed before launching the task body.
    BLOCKING = 1

    # Default events are recorded for each stream in the task.
    # The host launches the task body without waiting for event dependencies to be completed.
    # The tasks work is enqueued on streams that wait for the event dependencies to be completed.
    NON_BLOCKING = 2

    # Default events are recorded for each stream in the task.
    # The runtime does not handle any synchronization around the task body.
    # Data movement is assumed to happen on the tasks associated 'default' events.
    # TODO(wlr): This is very poorly supported and highly experimental.
    USER = 3


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
        self._context_stack = LocalStack()
        self._stream_stack = LocalStack()
        self._scheduler_stack = LocalStack()
        self._index = 0

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
