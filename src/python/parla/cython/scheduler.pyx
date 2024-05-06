# cython: language_level=3
# cython: language=c++
"""!
@file scheduler.pyx
@brief Contains the core Python logic to manage workers and launch tasks.
"""

from abc import abstractmethod
import threading
import inspect 
from ..common.globals import DeviceType, cupy, CUPY_ENABLED
from ..common.globals import SynchronizationType as SyncType
from ..common.globals import _Locals as Locals 
from ..common.globals import USE_PYTHON_RUNAHEAD, _global_data_tasks, PREINIT_THREADS

if cupy is not None:
    import cupy_backends
else:
    cupy_backends = None


import traceback

from . import tasks

from . cimport core
from . import core
from .cyparray import CyPArray
from ..utility.tracer import NVTXTracer

Task = tasks.Task
ComputeTask = tasks.ComputeTask
DataMovementTask = tasks.DataMovementTask
TaskSpace = tasks.TaskSpace
AtomicTaskSpace = tasks.AtomicTaskSpace
create_env = tasks.create_env

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

                device_manager = self.scheduler.device_manager

                while self._should_run:
                    self.status = "Waiting"
                    self.inner_worker.wait_for_task()
                    self.task = self.inner_worker.get_task()
                    
                    if isinstance(self.task, core.DataMovementTaskAttributes):
                        self.task_attrs = self.task
                        self.task = DataMovementTask()
                        self.task.instantiate(self.task_attrs, self.scheduler)
                        self.task_attrs = None
                
                        # comment(wlr): Need this is all cases currently. FIXME: Add stream/event creation in C++ so python isn't the owner.
                        _global_data_tasks[id(self.task)] = self.task

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

                        # Push the task to the thread local stack
                        Locals.push_task(active_task)

                        with device_context as env:

                            if isinstance(active_task, ComputeTask):
                                # Perform write invalidations
                                for parray, target_idx in active_task.dataflow.inout:
                                    target_device = parla_devices[target_idx]
                                    global_target_id = target_device.get_global_id()
                                    parray_target_id = device_manager.globalid_to_parrayid(global_target_id)
                                    parray._auto_move(parray_target_id, True)
                            
                            # Run the task body (this may complete the task or return a continuation)
                            # The body may return asynchronusly before kernels have completed, in which case the task will be marked as runahead
                            active_task.run()
                            state = active_task.state

                        # Pop the task from the thread local stack
                        Locals.pop_task()

                        # Log events on all 'task default' streams
                        device_context.record_events()

                        # FIXME: This can be cleaned up and hidden from this function with a better interface...
                        if active_task.runahead == SyncType.NONE:
                            device_context.finalize()

                        if isinstance(state, tasks.TaskRunning):
                            
                            active_task.dependencies = state.dependencies
                            active_task.func = state.func
                            active_task.args = state.args

                            active_task.inner_task.clear_dependencies()
                            active_task.add_dependencies(active_task.dependencies, process=False)
                        
                        # print("Cleaning up Task", active_task, flush=True)
                        
                        if USE_PYTHON_RUNAHEAD:
                            # Handle synchronization in Python (for debugging, works!)
                            self.scheduler.inner_scheduler.task_cleanup_presync(self.inner_worker, active_task.inner_task, state.value)
                            if active_task.runahead != SyncType.NONE:
                                device_context.synchronize(events=True)
                            self.scheduler.inner_scheduler.task_cleanup_postsync(self.inner_worker, active_task.inner_task, state.value)
                        else:
                            # Handle synchronization in C++
                            self.scheduler.inner_scheduler.task_cleanup(self.inner_worker, active_task.inner_task, state.value)

                        if active_task.runahead != SyncType.NONE:
                            device_context.return_streams()

                        if active_task.is_completed():
                            active_task.cleanup()
                            active_task.state = tasks.TaskCompleted(active_task.result)

                        self.task = None

                    elif self._should_run:
                        raise WorkerThreadException("%r Worker: Woke without a task", self.index)
                    else:
                        break

        except Exception as e:
            tb = traceback.format_exc()
            print("Exception in Worker Thread ", self, " during handling of ", self.task.name, ": ", e, tb, flush=True)

            self.scheduler.exception_stack.append(e)
            self.scheduler.stop()

    def stop(self):
        super().stop()
        self.inner_worker.stop()


class Scheduler(ControllableThread, SchedulerContext):

    def __init__(self, memory_manager, device_manager, n_threads=6, period=0.001):
        super().__init__()

        self.start_monitor = threading.Condition(threading.Lock())

        self._monitor = threading.Condition(threading.Lock())

        self.exception_stack = []

        self.default_taskspace = AtomicTaskSpace("global")

        # TODO: Deprecate this
        resources = 1.0

        self.memory_manager = memory_manager
        self.device_manager = device_manager
        cy_memory_manager = self.memory_manager.get_cy_memory_manager()
        cy_device_manager = self.device_manager.get_cy_device_manager()
        self.inner_scheduler = PyInnerScheduler(cy_memory_manager,
                                                cy_device_manager,
                                                n_threads,
                                                resources, self)
        # Worker threads and a scheduler both can access the active_parrays
        # and so we need a lock to guard that.
        self.active_parrays_monitor = threading.Condition(threading.Lock())

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

            #print("Exiting Scheduler", flush=True)

        except Exception as e:
            self.exception_stack.append(e)
        finally:
            #print(self.exception_stack, flush=True)
            if len(self.exception_stack) > 0:
                raise self.exception_stack[0]
            pass

    def parray_eviction(self):
        py_mm = self.memory_manager
        # print("Eviction policy is activated")
        for cuda_device in self.device_manager.get_devices(DeviceType.CUDA):
            global_id = cuda_device.get_global_id()
            parray_id = self.device_manager.globalid_to_parrayid(global_id)
            # Get target memory size to evict from this device
            memory_size_to_evict = \
                self.inner_scheduler.get_memory_size_to_evict(global_id)
            # Get the number of PArray candidates that are allowed to be evicted
            # from Python eviction manager.
            num_evictable_parray = py_mm.size(global_id)
            # TODO(hc): remove this. this is for test.
            # import cupy
            for i in range(0, num_evictable_parray):
                try:
                    # Get a PArray from a memory manager to evict.
                    evictable_parray = \
                        py_mm.remove_and_return_head_from_zrlist(global_id)
                    if evictable_parray is not None:
                        evictable_parray.evict(parray_id)

                        # Repeat eviction until it gets enough memory.
                        memory_size_to_evict -= \
                            evictable_parray.nbytes_at(parray_id)
                        # print("\t Remaining size to evict:", memory_size_to_evict, flush=True)
                        if memory_size_to_evict <= 0:
                            break
                except Exception as e:
                    print("Failed to find parray evictable", flush=True)
        return

    def run(self):
        with self:
            while True:
                print("Scheduler: Running", flush=True)
                self.inner_scheduler.run()
                should_run = self.inner_scheduler.get_should_run()
                if should_run is False:
                    break
                # This case is executed if PArray eviction
                # mechanism was invoked by C++ scheduler.
                self.parray_eviction() 
            self.stop_callback()

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
        
    def create_parray(self, cy_parray: CyPArray, parray_dev_id: int):
        """
        Reserve PArray instances that are created through
        __init__() of the PArray class.
        In the current Parla, crosspy calls this function
        during initialization if its internal array type is PArray.

        :param parray: Created Cython PArray instance
        """
        self.inner_scheduler.create_parray(cy_parray, parray_dev_id)

    def get_mapped_memory(self, global_dev_id: int):
        """
        Return the total amount of mapped memory on a device.

        :param global_dev_id: global logical device id that
                              this function interests
        """
        return self.inner_scheduler.get_mapped_memory(global_dev_id)

    def get_reserved_memory(self, global_dev_id: int):
        """
        Return the total amount of reserved memory on a device.

        :param global_dev_id: global logical device id that
                              this function interests
        """
        return self.inner_scheduler.get_reserved_memory(global_dev_id)

    def get_max_memory(self, global_dev_id: int):
        """
        Return the total amount of memory on a device.

        :param global_dev_id: global logical device id that
                              this function interests
        """
        return self.inner_scheduler.get_max_memory(global_dev_id)

    def get_mapped_parray_state(self, global_dev_id: int, parray_parent_id):
        """
        Return True if a parent PArray of the passed PArray exists on a
        device.

        :param global_dev_id: global logical device id that 
                            this function interests 
        :param parray_parent_id: parent PArray ID
        """
        return self.inner_scheduler.get_mapped_parray_state(global_dev_id, parray_parent_id)

    def get_reserved_parray_state(self, global_dev_id: int, parray_parent_id):
        """
        Return True if a parent PArray of the passed PArray exists on a
        device.

        :param global_dev_id: global logical device id that 
                              this function interests 
        :param parray_parent_id: parent PArray ID
        """
        return self.inner_scheduler.get_parray_state(global_dev_id, parray_parent_id)

    def remove_parray_from_tracker(self, cy_parray: CyPArray, did: int):
        """
        Remove the evicted PArray instance on device `global_dev_id`
        from the PArray tracker's table

        :param cy_parray: Cython PArray instance to be removed 
        :param did: global logical device id where the PArray is evicted
        """
        self.inner_scheduler.remove_parray_from_tracker(cy_parray, did)


cpdef _task_callback(task, body):
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
                if in_value_task is not None:
                    in_value = in_value_task.result
                    #print(in_value_task.state)
                    #print(f"Task invalue1", task, in_value_task, body, in_value, in_value_task.state, in_value_task.result, type(in_value_task), flush=True)
                else:
                    in_value = None
                    #print(f"Task invalue2", task, in_value_task, body, in_value, type(task), type(in_value_task), flush=True)
                
                new_task_info = body.send(in_value)
                #print(f"Task new_task_info", task, new_task_info, body, flush=True)
                task.value_task = None
                if not isinstance(new_task_info, tasks.TaskAwaitTasks):
                    raise TypeError(
                        "Parla coroutine tasks must yield a TaskAwaitTasks")
                dependencies = new_task_info.dependencies
                value_task = new_task_info.value_task

                if value_task:
                    assert isinstance(value_task, Task)
                    task.value_task = value_task
                return tasks.TaskRunning(_task_callback, (body,), dependencies, id=task.name)
            except StopIteration as e:
                #print(f"Task StopIteration", task, e, e.args, flush=True)
                result = None
                if e.args:
                    (result,) = e.args
                return tasks.TaskRunahead(result)
        else:
            result = body()
            #print(f"Task body", task, body, result, flush=True)
            return tasks.TaskRunahead(result)
    finally:
        pass
