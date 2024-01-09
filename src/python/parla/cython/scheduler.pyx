# cython: language_level=3
# cython: language=c++
"""!
@file scheduler.pyx
@brief Contains the core Python logic to manage workers and launch tasks.
"""

from abc import abstractmethod
import threading
import inspect 
from parla.common.globals import DeviceType, cupy, CUPY_ENABLED
from parla.common.globals import SynchronizationType as SyncType

if cupy is not None:
    import cupy_backends
else:
    cupy_backends = None

import traceback

from . import tasks

from . cimport core
from . import core
from .cyparray import CyPArray

from ..common.globals import _Locals as Locals 
from ..common.globals import USE_PYTHON_RUNAHEAD, _global_data_tasks, PREINIT_THREADS

Task = tasks.Task
ComputeTask = tasks.ComputeTask
DataMovementTask = tasks.DataMovementTask
TaskSpace = tasks.TaskSpace
AtomicTaskSpace = tasks.AtomicTaskSpace
create_env = tasks.create_env

from parla.utility.tracer import NVTXTracer

PyInnerScheduler = core.PyInnerScheduler
PyInnerWorker = core.PyInnerWorker
PyInnerTask = core.PyInnerTask

nvtx = NVTXTracer
nvtx.initialize()


class TaskBodyException(RuntimeError):
    pass


class SchedulerException(RuntimeError):
    pass


class WorkerThreadException(RuntimeError):
    pass


class _SchedulerLocals(threading.local):
    def __init__(self):
        super(_SchedulerLocals, self).__init__()
        self._scheduler_context_stack = []

    @property
    def scheduler_context(self):
        if self._scheduler_context_stack:
            return self._scheduler_context_stack[-1]
        else:
            raise Exception("No scheduler context")


_scheduler_locals = _SchedulerLocals()


def get_scheduler_context():
    return _scheduler_locals.scheduler_context


def get_device_manager():
    return get_scheduler_context().device_manager


def get_stream_pool():
    return get_scheduler_context().device_manager.stream_pool


class SchedulerContext:

    # TODO: Add enviornments back

    @property
    @abstractmethod
    def scheduler(self) -> "Scheduler":
        raise NotImplementedError()

    def __enter__(self):
        # TODO: Deprecate _scheduler_locals 
        _scheduler_locals._scheduler_context_stack.append(self)
        Locals.push_scheduler(self.scheduler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _scheduler_locals._scheduler_context_stack.pop()
        Locals.pop_scheduler()


class ControllableThread(threading.Thread):

    def __init__(self):
        super().__init__()
        self._should_run = True

    def stop(self):
        # print("Stopping Thread:", self, flush=True)
        with self._monitor:
            self._should_run = False
            self._monitor.notify_all()

    @abstractmethod
    def run(self):
        pass


class WorkerThread(ControllableThread, SchedulerContext):
    def __init__(self, scheduler, index):
        super().__init__()
        self._scheduler = scheduler
        self._monitor = threading.Condition(threading.Lock())

        self.index = index

        self.task = None
        self.status = "Initializing"

        self.inner_worker = PyInnerWorker(self, scheduler.inner_scheduler)

        # Add the worker to the scheduler pool of all workers (not yet active)
        scheduler.inner_scheduler.add_worker(self.inner_worker)

    def start(self, initialize=True):
        super(ControllableThread, self).start()

        if PREINIT_THREADS:
            self._initialize()

    def _initialize(self):
        device_manager = self.scheduler.device_manager

        # comment(wlr): wow, it is non-trivial to get the set of active cuda devices in the scheduler.
        # TODO(wlr): Fix this in device_manager (see todo there)

        if CUPY_ENABLED:
            gpu_arch = device_manager.py_registered_archs[DeviceType.GPU]
            ngpus = len(gpu_arch)

            for index in range(ngpus):
                # Trigger cuBLAS/etc. initialization for this GPU in this thread.
                with cupy.cuda.Device(index % device_manager.num_real_gpus) as device:
                    a = cupy.asarray([2.])
                    cupy.cuda.get_current_stream().synchronize()
                    with cupy.cuda.Stream(False, True) as stream:
                        cupy.asnumpy(cupy.sqrt(a))
                        device.cublas_handle
                        device.cusolver_handle

                        try:
                            device.cusolver_sp_handle
                        except cupy_backends.cuda.libs.cusolver.CUSOLVERError:
                            pass

                        device.cusparse_handle
                        
                        stream.synchronize()
                        device.synchronize()

    @property
    def scheduler(self):
        return self._scheduler

    def assign_task(self, task):
        with self._monitor:
            if self.task:
                raise Exception("Worker already has a task")
            self.task = task
            self._monitor.notify()

    def remove_task(self):
        with self._monitor:
            if not self.task:
                raise Exception("Worker does not have a task")
            self.task = None

    def run(self):
        try:
            # A worker thread is a scheduler context
            with self:
                # TODO: Perform any thread initialization on enviornment components

                # Add the worker to the scheduler pool of active & availabe workers
                self.scheduler.inner_scheduler.enqueue_worker(self.inner_worker)

                with self.scheduler.start_monitor:
                    self.scheduler.start_monitor.notify_all()

                while self._should_run:
                    self.status = "Waiting"

                    nvtx.push_range(message="worker::wait", domain="Python Runtime", color="blue")
                    self.inner_worker.wait_for_task()

                    self.task = self.inner_worker.get_task()
                    if isinstance(self.task, core.DataMovementTaskAttributes):
                        self.task_attrs = self.task
                        self.task = DataMovementTask()
                        self.task.instantiate(self.task_attrs, self.scheduler)
                        self.task_attrs = None
                
                        # comment(wlr): Need this is all cases currently. FIXME: Add stream/event creation in C++ so python isn't the owner.
                        _global_data_tasks[id(self.task)] = self.task

                    nvtx.pop_range(domain="Python Runtime")

                    self.status = "Running"

                    if isinstance(self.task, Task):
                        active_task = self.task 

                        parla_devices = active_task.get_assigned_devices()
                        device_context = create_env(parla_devices)

                        # Save device_context with task object
                        active_task.environment = device_context

                        # Writes all 'default' streams and event pointers to c++ task
                        device_context.write_to_task(active_task)

                        # Wait / Enqueue event for dependencies to complete
                        if USE_PYTHON_RUNAHEAD:
                            active_task.py_handle_runahead_dependencies() 
                        else:
                            active_task.handle_runahead_dependencies()

                        nvtx.push_range(message="worker::run", domain="Python Runtime", color="blue")

                        # Push the task to the thread local stack
                        Locals.push_task(active_task)

                        with device_context as env:
                            
                            core.binlog_2("Worker", "Running task: ", active_task.inner_task, " on worker: ", self.inner_worker)
                            # Run the task body (this may complete the task or return a continuation)
                            # The body may return asynchronusly before kernels have completed, in which case the task will be marked as runahead
                            active_task.run()

                        # Pop the task from the thread local stack
                        Locals.pop_task()

                        # Log events on all 'task default' streams
                        device_context.record_events()

                        nvtx.pop_range(domain="Python Runtime")

                        nvtx.push_range(message="worker::cleanup", domain="Python Runtime", color="blue")

                        final_state = active_task.state

                        # FIXME: This can be cleaned up and hidden from this function with a better interface...
                        if active_task.runahead == SyncType.NONE:
                            device_context.finalize()

                        # TODO(wlr): Add better exception handling
                        if isinstance(final_state, tasks.TaskException):
                            raise TaskBodyException(active_task.state.exception)

                        elif isinstance(final_state, tasks.TaskRunning):
                            nvtx.push_range(message="worker::continuation", domain="Python Runtime", color="red")
                            # print("CONTINUATION: ", active_task.taskid.full_name, active_task.state.dependencies, flush=True)
                            active_task.dependencies = active_task.state.dependencies
                            active_task.func = active_task.state.func
                            active_task.args = active_task.state.args

                            active_task.inner_task.clear_dependencies()
                            active_task.add_dependencies(active_task.dependencies, process=False)
                            nvtx.pop_range(domain="Python Runtime")
                        
                        elif  isinstance(final_state, tasks.TaskRunahead):
                            core.binlog_2("Worker", "Runahead task: ", active_task.inner_task, " on worker: ", self.inner_worker)
                    
                        # print("Cleaning up Task", active_task, flush=True)
                        
                        if USE_PYTHON_RUNAHEAD:
                            # Handle synchronization in Python (for debugging, works!)
                            self.scheduler.inner_scheduler.task_cleanup_presync(self.inner_worker, active_task.inner_task, active_task.state.value)
                            if active_task.runahead != SyncType.NONE:
                                device_context.synchronize(events=True)
                            self.scheduler.inner_scheduler.task_cleanup_postsync(self.inner_worker, active_task.inner_task, active_task.state.value)
                        else:
                            # Handle synchronization in C++
                            self.scheduler.inner_scheduler.task_cleanup(self.inner_worker, active_task.inner_task, active_task.state.value)

                        if active_task.runahead != SyncType.NONE:
                            device_context.return_streams()

                        if isinstance(final_state, tasks.TaskRunahead):
                            final_state = tasks.TaskCompleted(final_state.return_value)
                            active_task.cleanup()

                            core.binlog_2("Worker", "Completed task: ", active_task.inner_task, " on worker: ", self.inner_worker)

                        active_task.state = final_state
                        self.task = None

                        nvtx.pop_range(domain="Python Runtime")
                    elif self._should_run:
                        raise WorkerThreadException("%r Worker: Woke without a task", self.index)
                    else:
                        break

        except Exception as e:
            tb = traceback.format_exc()
            print("Exception in Worker Thread ", self, ": ", e, tb, flush=True)

            self.scheduler.exception_stack.append(e)
            self.scheduler.stop()

            if isinstance(e, TaskBodyException):
                raise WorkerThreadException(f"Unhandled Exception in Task: {self.task.get_name()}") from e
            if isinstance(e, KeyboardInterrupt):
                print("You pressed Ctrl+C! In a worker!", flush=True)
                raise e
            else:
                raise WorkerThreadException("Unhandled Exception on "+str(self))

    def stop(self):
        super().stop()
        self.inner_worker.stop()


class Scheduler(ControllableThread, SchedulerContext):

    def __init__(self, device_manager, n_threads=6, period=0.001):
        super().__init__()

        self.start_monitor = threading.Condition(threading.Lock())

        self._monitor = threading.Condition(threading.Lock())

        self.exception_stack = []

        self.default_taskspace = AtomicTaskSpace("global")

        # TODO: Deprecate this
        resources = 1.0

        self.device_manager = device_manager
        cy_device_manager = self.device_manager.get_cy_device_manager()
        self.inner_scheduler = PyInnerScheduler(cy_device_manager, n_threads, resources, self)

        self.worker_threads = [WorkerThread(self, i) for i in range(n_threads)]

        with self.start_monitor:
            for thread in self.worker_threads:
                thread.start()
            self.start_monitor.wait()

        self.start()

    @property
    def scheduler(self):
        return self

    def get_device_reqs_from_placement(self, placement, vcus, memory):
        return self.device_manager.get_device_reqs_from_placement(placement, vcus, memory)

    def __enter__(self):
        if self.inner_scheduler.get_num_active_tasks() != 1:
            raise SchedulerException("Schedulers can only have a single scope.")
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            super().__exit__(exc_type, exc_val, exc_tb)
            self.inner_scheduler.decrease_num_active_tasks()

            with self._monitor:
                while self.inner_scheduler.get_status():
                    self._monitor.wait()

                for t in self.worker_threads:
                    t.join()
        except Exception as e:
            self.exception_stack.append(e)

            if len(self.exception_stack) > 0:
                raise self.exception_stack[0]
        finally:
            pass

    def run(self):
        self.inner_scheduler.run()

    def stop(self):
        self.inner_scheduler.stop()

    def get_num_running_tasks(self):
        return self.inner_scheduler.get_num_running_tasks()

    def stop_callback(self):
        super().stop()

        for w in self.worker_threads:
            w.stop()

    def spawn_task(self, task):
        self.inner_scheduler.spawn_task(task.inner_task)

    def assign_task(self, task, worker):
        task.state = tasks.TaskRunning(task.func, task.args, task.dependencies)
        worker.assign_task(task)

    def get_num_notified_workers(self):
        return self.inner_scheduler.get_num_notified_workers()

    def spawn_wait(self):
        self.inner_scheduler.spawn_wait()

    def reserve_parray(self, cy_parray: CyPArray, global_dev_id: int):
        """
        Reserve PArray instances that are created through
        __init__() of the PArray class.
        In the current Parla, crosspy calls this function
        during initialization if its internal array type is PArray.

        :param parray: Created Cython PArray instance
        :param global_dev_id: global logical device id that
                              the PArray will be placed
        """
        self.inner_scheduler.reserve_parray(cy_parray, global_dev_id)

    def release_parray(self, cy_parray: CyPArray, global_dev_id: int):
        """
        Release PArray instances that are evicted.

        :param parray: Cython PArray instance to be evicted
        :param global_dev_id: global logical device id that
                              the PArray will be evicted
        """
        self.inner_scheduler.release_parray(cy_parray, global_dev_id)

    def get_parray_state(self, global_dev_id: int, parray_parent_id):
        """
        Return True if a parent PArray of the passed PArray exists on a
        device.

        :param global_dev_id: global logical device id that 
                              this function interests 
        :param parray_parent_id: parent PArray ID
        """
        return self.inner_scheduler.get_parray_state(global_dev_id, parray_parent_id)


def _task_callback(task, body):
    """
    A function which forwards to a python function in the appropriate device context.
    """
    try:
        body = body

        if inspect.iscoroutinefunction(body):
            body = body()

        if inspect.iscoroutine(body):
            try:
                in_value_task = getattr(task, "value_task", None)
                in_value = in_value_task and in_value_task.result

                new_task_info = body.send(in_value)
                task.value_task = None
                if not isinstance(new_task_info, tasks.TaskAwaitTasks):
                    raise TypeError(
                        "Parla coroutine tasks must yield a TaskAwaitTasks")
                dependencies = new_task_info.dependencies
                value_task = new_task_info.value_task
                if value_task:
                    assert isinstance(value_task, Task)
                    task.value_task = value_task
                return tasks.TaskRunning(_task_callback, (body,), dependencies)
            except StopIteration as e:
                result = None
                if e.args:
                    (result,) = e.args
                return tasks.TaskRunahead(result)
        else:
            result = body()
            return tasks.TaskRunahead(result)
    finally:
        pass
