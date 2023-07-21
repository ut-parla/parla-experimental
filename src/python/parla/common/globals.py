from __future__ import annotations # For type hints of unloaded classes

from enum import IntEnum
import threading
import os

try:
    import cupy
    num_gpu = cupy.cuda.runtime.getDeviceCount()
    CUPY_ENABLED = (os.getenv("PARLA_ENABLE_CUPY", "1") == "1")
except Exception as e:
    cupy = None
    CUPY_ENABLED = False

if CUPY_ENABLED:
    try:
        import crosspy
        CROSSPY_ENABLED = (os.getenv("PARLA_ENABLE_CROSSPY", "1") == "1")
    except ImportError:
        CROSSPY_ENABLED = False
        crosspy = None
else:
    CROSSPY_ENABLED = False
    crosspy = None


USE_PYTHON_RUNAHEAD = (os.getenv("PARLA_ENABLE_PYTHON_RUNAHEAD", "1") == "1")
PREINIT_THREADS = (os.getenv("PARLA_PREINIT_THREADS", "1") == "1")

print("USE_PYTHON_RUNAHEAD: ", USE_PYTHON_RUNAHEAD)
print("CUPY_ENABLED: ", CUPY_ENABLED)
print("PREINIT_THREADS: ", PREINIT_THREADS)

_global_data_tasks = {}


VCU_BASELINE=1000


class SynchronizationType(IntEnum):
    """
    This class declares the type (if any) of runeahead synchronization
    """
    NONE = 0
    BLOCKING = 1
    NON_BLOCKING = 2
    USER = 3


SYNC_FLAG = os.getenv("PARLA_DEFAULT_SYNC", "0")

if SYNC_FLAG == "0":
    default_sync = SynchronizationType.NONE
elif SYNC_FLAG == "1":
    default_sync = SynchronizationType.BLOCKING
else:
    default_sync = SynchronizationType.NON_BLOCKING

print("DEFAULT SYNC: ", default_sync)


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


class Storage():

    # This is literally just a dictionary wrapper.
    # It's here to make it easier to swap out the storage implementation later or error handling.
    # Not sure if necessary, but it's here for now.

    def __init__(self):
        self._store = {}

    def store(self, key, value):
        self._store[key] = value

    def retrieve(self, key):
        return self._store[key]

    def clear(self):
        self._store = {}

    def __repr__(self):
        return str(self._store)

    def __str__(self):
        return self.__repr__()

    def __contains__(self, key):
        return key in self._store

    def __getitem__(self, key):
        return self._store[key]


# TODO(wlr): Move this to the scheduler context.
_GlobalStorage = Storage()


class LocalStorage(threading.local, Storage):
    def __init__(self):
        super(LocalStorage, self).__init__()


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
        self._store = LocalStorage()
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

    def store(self, key, value):
        self._store.store(key, value)

    def retrieve(self, key):
        return self._store.retrieve(key)

    @property
    def storage(self):
        return self._store


_Locals = Locals()


def get_locals():
    return _Locals

def get_current_task():
    return _Locals.task

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
    scheduler = get_scheduler()
    if scheduler is None:
        raise RuntimeError("Attempted to access device manager, but no scheduler is active.")
    else:
        return scheduler.device_manager

def get_stream_pool():
    return get_device_manager().stream_pool


def has_environment():
    return True if _Locals.context else False
