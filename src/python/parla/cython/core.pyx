import cython 

from parla.common.parray.core import PArray
from parla.common.dataflow import Dataflow
from parla.common.globals import AccessMode

from parla.cython.device cimport Device
from parla.cython.cyparray cimport CyPArray
from parla.cython.device_manager cimport CyDeviceManager, DeviceManager
import threading
from enum import IntEnum, auto
from parla.common.globals import cupy
from libc.stdint cimport uintptr_t

#Resource Types
#TODO: Python ENUM

#Logging functions

LOG_TRACE = 0
LOG_DEBUG = 1
LOG_INFO = 2
LOG_WARN = 3
LOG_ERROR = 4
LOG_FATAL = 5

cpdef py_write_log(filename):
    fname = filename.encode('utf-8')
    write_log(fname)

cpdef py_init_log(filename):
    fname = filename.encode('utf-8')
    initialize_log(fname)

cpdef _log_task(logging_level, category, message, PyInnerTask obj):
    cdef InnerTask* _inner = <InnerTask*> obj.c_task
    msg = message.encode('utf-8')

    if category == "Task":
        log_task_1[InnerTask](logging_level, msg, _inner)
    if category == "Worker":
        log_worker_1[InnerTask](logging_level, msg, _inner)
    if category == "Scheduler":
            log_scheduler_1[InnerTask](logging_level, msg, _inner)

cpdef _log_worker(logging_level, category, message, PyInnerWorker obj):
    cdef InnerWorker* _inner = <InnerWorker*> obj.inner_worker
    msg = message.encode('utf-8')

    if category == "Task":
        log_task_1[InnerWorker](logging_level, msg, _inner)
    elif category == "Worker":
        log_worker_1[InnerWorker](logging_level, msg, _inner)
    elif category == "Scheduelr":
        log_scheduler_1[InnerWorker](logging_level, msg, _inner)

cpdef _log_task_worker(logging_level, category, message1, PyInnerTask obj1, message2, PyInnerWorker obj2):
    cdef InnerTask* _inner1 = <InnerTask*> obj1.c_task
    cdef InnerWorker* _inner2 = <InnerWorker*> obj2.inner_worker
    msg1 = message1.encode('utf-8')
    msg2 = message2.encode('utf-8')

    if  category == "Task":
        log_task_2[InnerTask, InnerWorker](logging_level, msg1, _inner1, msg2, _inner2)
    if category  == "Worker":
        log_worker_2[InnerTask, InnerWorker](logging_level, msg1, _inner1, msg2, _inner2)
    if category == "Scheduler":
        log_scheduler_2[InnerTask, InnerWorker](logging_level, msg1, _inner1, msg2, _inner2)

inner_type1  = cython.fused_type(PyInnerTask, PyInnerWorker)
inner_type2 = cython.fused_type(PyInnerTask, PyInnerWorker)

cpdef binlog_0(category, message, logging_level=LOG_INFO):
    msg = message.encode('utf-8')
    if  category == "Task":
        log_task_msg(logging_level, msg)
    if category  == "Worker":
        log_worker_msg(logging_level, msg)
    if category == "Scheduler":
        log_scheduler_msg(logging_level, msg)

cpdef binlog_1(category, message, inner_type1 obj, logging_level=LOG_INFO):

    if inner_type1 is PyInnerTask:
        _log_task(logging_level, category, message, obj)
    elif inner_type1 is PyInnerWorker:
        _log_worker(logging_level, category, message, obj)
    else:
        raise Exception("Unknown type in logger function")

cpdef binlog_2(category, message1, inner_type1 obj1, message2, inner_type2 obj2, logging_level=LOG_INFO):

    if inner_type1 is PyInnerTask:
        if inner_type2 is PyInnerWorker:
            _log_task_worker(logging_level, category, message1, obj1, message2, obj2)
        else:
            raise Exception("Unknown type combination in logger function")
    else:
        raise Exception("Unknown type combination in logger function")

#cpdef log_2(category, message1, inner_type1  obj1,  message2, inner_type2 obj2):



cpdef cpu_bsleep_gil(unsigned int microseconds):
    """Busy sleep for a given number of microseconds, but don't release the GIL"""
    cpu_busy_sleep(microseconds)

cpdef cpu_bsleep_nogil(unsigned int microseconds):
    """Busy sleep for a given number of microseconds, but release the GIL"""
    with nogil:
        cpu_busy_sleep(microseconds)

cpdef gpu_bsleep_gil(dev, t, stream):
    cdef int c_dev = dev
    cdef unsigned long c_t = t
    cdef uintptr_t c_stream = stream.ptr
    gpu_busy_sleep(c_dev, c_t, c_stream)

cpdef gpu_bsleep_nogil(dev, t, stream):
    cdef int c_dev = dev
    cdef unsigned long c_t = t
    cdef uintptr_t c_stream = stream.ptr
    with nogil:
        gpu_busy_sleep(c_dev, c_t, c_stream)

# Define callbacks for C++ to call back into Python

cdef void callback_launch(void* python_scheduler, void* python_task, void*
        python_worker) nogil:
    with gil:
        #print("Inside callback to cython", flush=True)
        task = <object>python_task
        scheduler = <object>python_scheduler
        worker = <object>python_worker

        scheduler.assign_task(task, worker)

        #print("Done with callback", flush=True)
        #(<object>python_function)(<object>python_input)

cdef void callback_stop(void* python_function) nogil:
    with gil:
        #print("Inside callback to cython (stop)", flush=True)
        scheduler = <object>python_function
        scheduler.stop_callback()

        #(<object>python_function)(<object>python_input)

#Define the Cython Wrapper Classes

cdef class PyInnerTask:
    cdef InnerTask* c_task
    cdef string name

    def __cinit__(self):
        cdef InnerTask* _c_task
        _c_task = new InnerTask()
        self.c_task = _c_task

    def __init__(self, long long int idx, object python_task):
        cdef InnerTask* _c_task
        _c_task = self.c_task

        binlog_1("Task", "Creating task", self)

        _c_task.set_id(idx)
        _c_task.set_py_task(<void *> python_task)

    cpdef set_py_task(self, python_task):
        cdef InnerTask* _c_task = self.c_task
        _c_task.set_py_task(<void*> python_task)

    cpdef update_name(self, string name):
        cdef InnerTask* _c_task = self.c_task
        self.name = name
        _c_task.set_name(name)

    def __dealloc__(self):
        binlog_0("Task", "Task {} is being deallocated".format(self.name))
        del self.c_task

    cpdef add_priority(self, priority):
        cdef InnerTask* _c_task = self.c_task
        cdef int c_priority = priority
        _c_task.set_priority(c_priority)

    cpdef add_constraints(self, vcus):
        cdef InnerTask* _c_task = self.c_task
        resource_type = "vcus"
        resource_type = resource_type.encode('utf-8')
        cdef float c_vcus = vcus
        _c_task.set_resources(resource_type, c_vcus)

    cpdef get_py_task(self):
        cdef InnerTask* c_self = self.c_task
        return <object> c_self.get_py_task()

    cpdef set_scheduler(self, PyInnerScheduler scheduler):
        cdef InnerTask* c_self = self.c_task
        cdef InnerScheduler* c_scheduler = scheduler.inner_scheduler
        c_self.set_scheduler(c_scheduler)

    cpdef add_dependencies(self, dependency_list, process=False):
        cdef InnerTask* c_self = self.c_task

        cdef PyInnerTask dependency
        cdef InnerTask* c_dependency

        cdef bool status = False 
        cdef _StatusFlags status_flags

        try: 
            for i in range(0, len(dependency_list)):
                d = dependency_list[i]
                dependency = d.inner_task
                c_dependency = dependency.c_task
                c_self.queue_dependency(c_dependency)
        except TypeError:
            for d in dependency_list:
                dependency = d.inner_task
                c_dependency = dependency.c_task
                c_self.queue_dependency(c_dependency)
                
        if process:
            with nogil:
                status_flags = c_self.process_dependencies()
                status = status_flags.mappable

        return status

    cpdef clear_dependencies(self):
        cdef InnerTask* c_self = self.c_task
        c_self.clear_dependencies()

    cpdef get_dependencies(self):
        cdef InnerTask* c_self = self.c_task

        cdef vector[void*] c_dependencies = c_self.get_dependencies()
        cdef size_t num_deps = c_dependencies.size()

        cdef PyInnerTask py_dependency
        cdef InnerTask* c_dependency

        dependencies = []
        for i in range(num_deps):
            c_dependency = <InnerTask*> c_dependencies[i]
            print("converting to python task: ")
            py_dependency = <PyInnerTask> c_dependency.get_py_task()
            dependencies.append(py_dependency)

        return dependencies

    cpdef get_dependents(self):
        cdef InnerTask* c_self = self.c_task

        cdef vector[void*] c_dependents = c_self.get_dependents()
        cdef size_t num_deps = c_dependents.size()

        cdef PyInnerTask py_dependent
        cdef InnerTask* c_dependent

        dependents = []
        for i in range(num_deps):
            c_dependent = <InnerTask*> c_dependents[i]
            py_dependent = <PyInnerTask> c_dependent.get_py_task()
            dependents.append(py_dependent)

        return dependents

    cpdef get_num_dependencies(self):
        cdef InnerTask* c_self = self.c_task
        return c_self.get_num_dependencies()

    cpdef get_num_dependents(self):
        cdef InnerTask* c_self = self.c_task
        return c_self.get_num_dependents()

    cpdef get_num_blocking_dependencies(self):
        cdef InnerTask* c_self = self.c_task
        return c_self.get_num_blocking_dependencies()

    cpdef get_num_unmapped_dependencies(self):
        cdef InnerTask* c_self = self.c_task
        return c_self.get_num_unmapped_dependencies()

    cpdef get_assigned_devices(self):
        cdef InnerTask* c_self = self.c_task

        cdef vector[Device*] c_devices = c_self.get_assigned_devices()
        cdef size_t num_devices = c_devices.size()

        cdef Device* c_device

        devices = []
        for i in range(num_devices):
            c_device = <Device*> c_devices[i]
            py_device = <object> c_device.get_py_device()
            devices.append(py_device)

        return devices

    cpdef add_parray(self, CyPArray cy_parray, flag, int dev_id):
        cdef InnerTask* c_self = self.c_task
        c_self.add_parray(cy_parray.get_cpp_parray(), int(flag), dev_id)

    cpdef notify_dependents_wrapper(self):
        cdef InnerTask* c_self = self.c_task
        cdef bool status = False
        with nogil:
            status = c_self.notify_dependents_wrapper()
        return status

    cpdef set_state(self, int state):
        cdef InnerTask* c_self = self.c_task
        return c_self.set_state(state)

    cpdef set_complete(self):
        cdef InnerTask* c_self = self.c_task
        c_self.set_state(7)

    cpdef add_device_req(self, CyDevice cy_device, long mem_sz, int num_vcus):
        cdef InnerTask* c_self = self.c_task
        cdef Device* cpp_device = cy_device.get_cpp_device()
        c_self.add_device_req(cpp_device, mem_sz, num_vcus)

    cpdef begin_arch_req_addition(self):
        cdef InnerTask* c_self = self.c_task
        c_self.begin_arch_req_addition()

    cpdef end_arch_req_addition(self):
        cdef InnerTask* c_self = self.c_task
        c_self.end_arch_req_addition()

    cpdef begin_multidev_req_addition(self):
        cdef InnerTask* c_self = self.c_task
        c_self.begin_multidev_req_addition()

    cpdef end_multidev_req_addition(self):
        cdef InnerTask* c_self = self.c_task
        c_self.end_multidev_req_addition()

    cpdef set_c_task(self, CyDataMovementTaskAttributes c_attrs):
        self.c_task = c_attrs.get_c_task()
        
    cpdef add_stream(self, py_stream):
        cdef uintptr_t i_stream 
        cdef InnerTask* c_self = self.c_task

        if isinstance(py_stream, cupy.cuda.Stream):
            i_stream = <uintptr_t> py_stream.ptr
            c_self.add_stream(i_stream)

    cpdef add_event(self, py_event):
        cdef uintptr_t i_event 
        cdef InnerTask* c_self = self.c_task

        if isinstance(py_event, cupy.cuda.Event):
            i_event = <uintptr_t> py_event.ptr
            c_self.add_event(i_event)

    cpdef reset_events_streams(self):
        cdef InnerTask* c_self = self.c_task
        c_self.reset_events_streams()

    cpdef handle_runahead_dependencies(self):
        cdef InnerTask* c_self = self.c_task
        c_self.handle_runahead_dependencies()

    cpdef synchronize_events(self):
        cdef InnerTask* c_self = self.c_task
        c_self.synchronize_events()
        


cdef class PyInnerWorker:
    cdef InnerWorker* inner_worker

    def __cinit__(self):
        cdef InnerWorker* _inner_worker
        _inner_worker = new InnerWorker()
        self.inner_worker = _inner_worker

    def __init__(self, python_worker, PyInnerScheduler python_scheduler):
        cdef InnerWorker* _inner_worker
        _inner_worker = self.inner_worker

        _inner_worker.set_py_worker(<void *> python_worker)
        _inner_worker.set_thread_idx(python_worker.index)

        cdef InnerScheduler* c_scheduler
        c_scheduler = python_scheduler.inner_scheduler
        _inner_worker.set_scheduler(c_scheduler)


    cpdef remove_task(self):
        cdef InnerWorker* _inner_worker
        _inner_worker = self.inner_worker

        _inner_worker.remove_task()

    cpdef wait_for_task(self):
        cdef InnerWorker* _inner_worker
        _inner_worker = self.inner_worker

        with nogil:
            _inner_worker.wait()

    cpdef get_task(self):
        cdef InnerWorker* _inner_worker
        _inner_worker = self.inner_worker

        cdef InnerTask* c_task
        cdef InnerDataTask* c_data_task
        cdef bool is_data_task = False
        cdef vector[Device*] c_devices 
        cdef size_t num_devices
        cdef Device* c_device

        if _inner_worker.ready:
            _inner_worker.get_task(&c_task, &is_data_task)
            if is_data_task == True:
                # This case is that the current task that
                # this worker thread gets is a data movement task.
                py_assigned_devices = []
                c_devices = c_task.get_assigned_devices()
                name = c_task.get_name()
                # Cast the base class instance to the inherited
                # data movement task.
                c_data_task = <InnerDataTask *> c_task
                num_devices = c_devices.size()
                # Construct a list of Python PArrays.
                for i in range(num_devices):
                    c_device = <Device *> c_devices[i]
                    py_device = <object> c_device.get_py_device()
                    py_assigned_devices.append(py_device)
                py_parray = <object> c_data_task.get_py_parray()
                access_mode = c_data_task.get_access_mode()
                dev_id = c_data_task.get_device_id();

                # Due to circular imports, the data movement task
                # is not created, but necessary information/objects
                # are created here.
                cy_data_attrs = CyDataMovementTaskAttributes()
                # A C++ pointer cannot be held in Python object.
                # Therefore, exploit a Cython class.
                cy_data_attrs.set_c_task(c_data_task)
                py_task = DataMovementTaskAttributes(name, py_parray, \
                                  access_mode, py_assigned_devices, cy_data_attrs, \
                                  dev_id)
            else:
                py_task = <object> c_task.get_py_task()
        else:
            py_task = None
        return py_task

    cpdef stop(self):
        cdef InnerWorker* _inner_worker
        _inner_worker = self.inner_worker

        _inner_worker.stop()

    def __dealloc__(self):
        del self.inner_worker

cdef class PyInnerScheduler:
    cdef InnerScheduler* inner_scheduler

    def __cinit__(self, CyDeviceManager cy_device_manager, int num_workers, float vcus, object python_scheduler):
        cdef InnerScheduler* _inner_scheduler
        cdef DeviceManager* _cpp_device_manager = <DeviceManager*> cy_device_manager.get_cpp_device_manager()

        _inner_scheduler = new InnerScheduler(_cpp_device_manager)
        self.inner_scheduler = _inner_scheduler

    def __init__(self, CyDeviceManager cy_device_manager, int num_workers, float vcus, object python_scheduler):
        cdef InnerScheduler* _inner_scheduler
        _inner_scheduler = self.inner_scheduler

        _inner_scheduler.set_num_workers(num_workers)

        resource_name = "vcus"
        resource_name = resource_name.encode('utf-8')
        _inner_scheduler.set_resources(resource_name, vcus)

        _inner_scheduler.set_py_scheduler(<void *> python_scheduler)

        cdef stopfunc_t py_stop = callback_stop
        _inner_scheduler.set_stop_callback(py_stop)

    cpdef get_status(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        return c_self.should_run

    def __dealloc__(self):
        del self.inner_scheduler

    cpdef run(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        with nogil:
            c_self.run()

    cpdef stop(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        c_self.stop()

    cpdef activate_wrapper(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        c_self.activate_wrapper()

    cpdef spawn_task(self, PyInnerTask task):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerTask* c_task = task.c_task

        c_self.spawn_task(c_task)

    cpdef add_worker(self, PyInnerWorker worker):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerWorker* c_worker = worker.inner_worker
        c_self.add_worker(c_worker)

    cpdef enqueue_worker(self, PyInnerWorker worker):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerWorker* c_worker = worker.inner_worker
        c_self.enqueue_worker(c_worker)

    #TODO(wlr): Should we release the GIL here? Or is it better to keep it?
    cpdef task_cleanup(self, PyInnerWorker worker, PyInnerTask task, int state):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerWorker* c_worker = worker.inner_worker
        cdef InnerTask* c_task = task.c_task
        with nogil:
            c_self.task_cleanup(c_worker, c_task, state)

    cpdef task_cleanup_presync(self, PyInnerWorker worker, PyInnerTask task, int state):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerWorker* c_worker = worker.inner_worker
        cdef InnerTask* c_task = task.c_task
        with nogil:
            c_self.task_cleanup_presync(c_worker, c_task, state)

    cpdef task_cleanup_postsync(self, PyInnerWorker worker, PyInnerTask task, int state):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerWorker* c_worker = worker.inner_worker
        cdef InnerTask* c_task = task.c_task
        with nogil:
            c_self.task_cleanup_postsync(c_worker, c_task, state)

    cpdef get_num_active_tasks(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        return c_self.get_num_active_tasks()

    cpdef increase_num_active_tasks(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        c_self.increase_num_active_tasks()

    cpdef decrease_num_active_tasks(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        c_self.decrease_num_active_tasks()

    #cpdef get_num_active_workers(self):
    #    cdef InnerScheduler* c_self = self.inner_scheduler
    #    return c_self.get_num_active_workers()

    cpdef get_num_ready_tasks(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        return c_self.get_num_ready_tasks()

    cpdef get_num_running_tasks(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        return c_self.get_num_running_tasks()

    cpdef get_num_notified_workers(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        return c_self.get_num_notified_workers()

    cpdef spawn_wait(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        with nogil:
            c_self.spawn_wait()

    cpdef reserve_parray(self, CyPArray cy_parray, int global_dev_id):
        cdef InnerScheduler* c_self = self.inner_scheduler
        c_self.reserve_parray(cy_parray.get_cpp_parray(), global_dev_id)

    cpdef release_parray(self, CyPArray cy_parray, int global_dev_id):
        cdef InnerScheduler* c_self = self.inner_scheduler
        c_self.release_parray(cy_parray.get_cpp_parray(), global_dev_id)

    cpdef get_parray_state(\
        self, int global_dev_id, long long int parray_parent_id):
        cdef InnerScheduler* c_self = self.inner_scheduler
        return c_self.get_parray_state(global_dev_id, parray_parent_id)


class Resources:

    def __init__(self, vcus):
        self.resources = vcus


cdef class CyDataMovementTaskAttributes:
    """
    While creating a Python data movement task,
    we need to connect the Python and the C++ instances.
    However, we cannot pass the C++ instance (or its pointer)
    through a normal Python class, but need to use a bridge
    Cython class. This is for that.
    """
    cdef InnerDataTask* c_data_task
    cdef set_c_task(self, InnerDataTask* c_data_task):
        self.c_data_task = c_data_task

    cdef InnerTask* get_c_task(self):
        return self.c_data_task


class DataMovementTaskAttributes:
    """
    Hold necessary information to create a data move task.
    This is delcared to avoid circular imports that could happen
    when we import tasks.pyx in here.
    """
    def __init__(self, name, py_parray: PArray, access_mode, assigned_devices, \
                 c_attrs: CyDataMovementTaskAttributes, dev_id):
        self.name = name
        self.parray = py_parray
        self.access_mode = access_mode
        self.assigned_devices = assigned_devices
        self.c_attrs = c_attrs
        self.dev_id = dev_id

