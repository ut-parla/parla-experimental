from parla.cython import tasks
from parla.cython import variants
from parla.common.spawn import spawn

from parla.common.globals import *


Tasks = tasks.TaskCollection
TaskSpace = tasks.TaskSpace
AtomicTaskSpace = tasks.AtomicTaskSpace
BackendTaskSpace = tasks.BackendTaskSpace

specialize = variants.specialize

__all__ = [
    spawn,
    Tasks,
    TaskSpace,
    AtomicTaskSpace,
    BackendTaskSpace,
    specialize,
    get_current_context,
    get_current_task,
    get_current_stream,
    get_current_devices,
]
