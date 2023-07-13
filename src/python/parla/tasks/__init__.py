from ..cython import tasks
from ..cython import variants
from ..common.spawn import spawn

from ..common.globals import *


Tasks = tasks.TaskCollection
TaskSpace = tasks.TaskSpace
AtomicTaskSpace = tasks.AtomicTaskSpace
BackendTaskSpace = tasks.BackendTaskSpace

specialize = variants.specialize

__all__ = [spawn, Tasks, TaskSpace, AtomicTaskSpace, BackendTaskSpace, specialize]
__all__ = __all__ + [get_current_context, get_current_task, get_current_stream, get_current_devices]
