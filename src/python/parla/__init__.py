import os
import sys
import signal
from .cython import tasks
from .cython import scheduler
from .cython import core
from .cython import device_manager
from .cython import device
from .cython import variants
from .common.spawn import spawn
from .common import parray
from .common.globals import PyMappingPolicyType

specialize = variants.specialize

sleep_gil = core.cpu_bsleep_gil
sleep_nogil = core.cpu_bsleep_nogil

gpu_sleep_gil = core.gpu_bsleep_gil
gpu_sleep_nogil = core.gpu_bsleep_nogil

TaskSpace = tasks.TaskSpace
Tasks = tasks.TaskCollection

DeviceManager = device_manager.PyDeviceManager

Stream = device.Stream
create_env = tasks.create_env
TaskEnvironment = tasks.TaskEnvironment
GPUEnvironment = tasks.GPUEnvironment


__all__ = ['spawn', 'TaskSpace', 'Parla', 'sleep_gil',
           'sleep_nogil', 'Tasks', 'parla_num_threads',
           'parray']


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

    def __init__(self,
                 mapping_policy: PyMappingPolicyType=PyMappingPolicyType.LoadBalancingLocality,
                 scheduler_class=scheduler.Scheduler,
                 sig_type=signal.SIGINT, logfile=None, n_workers=None,
                 dev_config_file=None, **kwds):
        assert issubclass(scheduler_class, scheduler.Scheduler)

        self.mapping_policy = mapping_policy
        self.scheduler_class = scheduler_class
        self.kwds = kwds
        self.sig = sig_type
        self.handle_interrupt = True
        self._device_manager = DeviceManager(dev_config_file)

        if logfile is None:
            logfile = os.environ.get("PARLA_LOGFILE", None)
        if logfile is None:
            logfile = "parla.blog"

        core.py_init_log(logfile)

        self.logfile = logfile
        if n_workers is None:
            n_workers = parla_num_threads

        self.kwds["n_threads"] = n_workers

    def __enter__(self):
        if hasattr(self, "_sched"):
            raise ValueError(
                "Do not use the same Parla object more than once.")
        self._sched = self.scheduler_class(self.mapping_policy, self._device_manager, **self.kwds)

        self.interuppted = False
        self.released = False

        try:
            self.original_handler = signal.getsignal(self.sig)

            def handler(signum, frame):
                print("YOU PRESSED CTRL+C, INTERRUPTING ALL TASKS", flush=True)
                self._sched.stop()
                self.release()
                self.interrupted = True

            signal.signal(self.sig, handler)
        except ValueError:
            # This happens if Parla is not running in the main thread.
            self.handle_interrupt = False
        finally:
            return self._sched.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return self._sched.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.release()
            del self._sched
            del self._device_manager
            # TODO(hc):This should not be necessary but without this,
            # cpu/gpu are not initialized even though Parla context
            # is completed and destroyed, and so if an outer loop
            # iterates more than 1 iterations, these are not initialized,
            # append devices while the device manager is initialized,
            # and finally, doubles the cpu/gpu devices.
            device_manager.cpu = device.PyCPUArchitecture()
            device_manager.gpu = device.PyCUDAArchitecture()
            core.py_write_log(self.logfile)

    def release(self):
        if self.released:
            return False

        try:
            if self.handle_interrupt:
                signal.signal(self.sig, self.original_handler)
        except ValueError:
            # This happens if Parla is not running in the main thread.
            pass
        finally:
            self.released = True
            return True
