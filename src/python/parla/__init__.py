from .cython import tasks
from .cython import scheduler
from .cython import core
from .common.spawn import spawn

from .common import containers

sleep_gil = core.cpu_bsleep_gil
sleep_nogil = core.cpu_bsleep_nogil

TaskSpace = containers.TaskSpace
Tasks = containers.Tasks

__all__ = ['spawn', 'TaskSpace', 'Parla', 'sleep_gil', 'sleep_nogil', 'Tasks', 'parla_num_threads']


import signal
import sys
import os

def signal_handler(signal, frame):
    print("You pressed Ctrl+C!")
    sys.exit(0)

parla_num_threads = os.environ.get("PARLA_NUM_THREADS", None)
if parla_num_threads is None:
    import psutil
    parla_num_threads = int(psutil.cpu_count(logical=False))
else:
    parla_num_threads = int(parla_num_threads)

class Parla:

    def __init__(self, scheduler_class=scheduler.Scheduler, sig_type=signal.SIGINT, logfile=None, n_workers=None, **kwds):
        assert issubclass(scheduler_class, scheduler.Scheduler)
        self.scheduler_class = scheduler_class
        self.kwds = kwds
        self.sig = sig_type

        if logfile is None:
            logfile = os.environ.get("PARLA_LOGFILE", None)
        if logfile is None:
            logfile = "parla.blog"

        self.logfile = logfile
        if n_workers is None:
            n_workers = parla_num_threads

        self.kwds["n_threads"] = n_workers


    def __enter__(self):
        if hasattr(self, "_sched"):
            raise ValueError(
                "Do not use the same Parla object more than once.")
        self._sched = self.scheduler_class(**self.kwds)

        self.interuppted = False
        self.released=False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            print("YOU PRESSED CTRL+C, INTERRUPTING ALL TASKS", flush=True)
            self._sched.stop()
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)

        return self._sched.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._sched.__exit__(exc_type, exc_val, exc_tb)
        finally:
            core.py_write_log(self.logfile)
            self.release()
            del self._sched

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)

        self.released = True

        return True