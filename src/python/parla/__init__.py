from .cython import tasks
from .cython import scheduler
from .cython import core
from .common.spawn import spawn

from .common import containers

sleep_gil = core.cpu_bsleep_gil
sleep_nogil = core.cpu_bsleep_nogil

TaskSpace = containers.TaskSpace

__all__ = ['spawn', 'TaskSpace', 'Parla', 'sleep_gil', 'sleep_nogil']


class Parla:

    def __init__(self, scheduler_class=scheduler.Scheduler, **kwds):
        assert issubclass(scheduler_class, scheduler.Scheduler)
        self.scheduler_class = scheduler_class
        self.kwds = kwds

    def __enter__(self):
        if hasattr(self, "_sched"):
            raise ValueError(
                "Do not use the same Parla object more than once.")
        self._sched = self.scheduler_class(**self.kwds)
        return self._sched.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._sched.__exit__(exc_type, exc_val, exc_tb)
        finally:
            del self._sched
