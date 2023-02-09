import nvtx
import cython 


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
    cdef string name #This is a hack to keep the name alive

    def __cinit__(self):
        cdef InnerTask* _c_task
        _c_task = new InnerTask()
        self.c_task = _c_task

    def __init__(self, long long int idx, object python_task, float vcus):
        cdef InnerTask* _c_task
        _c_task = self.c_task

        binlog_1("Task", "Creating task", self)

        if(python_task.taskid is not None):
            name = python_task.taskid.full_name
        else:
            name = python_task.name
            
        #name = "test"
        
        name = name.encode('utf-8')
        self.name = name
        _c_task.set_name(name)

        _c_task.set_id(idx)
        _c_task.set_py_task(<void *> python_task)

        priority = 0
        _c_task.set_priority(priority)

        resource_type = "vcus"
        resource_type = resource_type.encode('utf-8')
        _c_task.set_resources(resource_type, vcus)

    def __dealloc__(self):
        binlog_0("Task", "Task {} is being deallocated".format(self.name))
        del self.c_task

    cpdef get_py_task(self):
        cdef InnerTask* c_self = self.c_task
        return <object> c_self.get_py_task()

    cpdef add_dependencies(self, dependency_list, process=True):
        cdef InnerTask* c_self = self.c_task

        cdef PyInnerTask dependency
        cdef InnerTask* c_dependency

        cdef bool status = False 

        for d in dependency_list:
            dependency = d.inner_task
            c_dependency = dependency.c_task
            c_self.queue_dependency(c_dependency)


        if process:
            with nogil:
                status = c_self.process_dependencies()

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

    cpdef notify_dependents_wrapper(self):
        cdef InnerTask* c_self = self.c_task
        cdef bool status = False
        with nogil:
            status = c_self.notify_dependents_wrapper()
        return status

    cpdef set_state(self, int state):
        cdef InnerTask* c_self = self.c_task
        c_self.set_state(state)

    cpdef set_complete(self):
        cdef InnerTask* c_self = self.c_task
        cdef bool state = True
        c_self.set_complete(state)

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

        if _inner_worker.ready:
            c_task = _inner_worker.get_task()
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

    def __cinit__(self):
        cdef InnerScheduler* _inner_scheduler
        _inner_scheduler = new InnerScheduler()
        self.inner_scheduler = _inner_scheduler

    def __init__(self, int num_workers, float vcus, object python_scheduler):
        cdef InnerScheduler* _inner_scheduler
        _inner_scheduler = self.inner_scheduler

        _inner_scheduler.set_num_workers(num_workers)

        resource_name = "vcus"
        resource_name = resource_name.encode('utf-8')
        _inner_scheduler.set_resources(resource_name, vcus)

        _inner_scheduler.set_py_scheduler(<void *> python_scheduler)

        cdef stopfunc_t py_stop = callback_stop
        _inner_scheduler.set_stop_callback(py_stop)

        cdef launchfunc_t py_launch = callback_launch
        _inner_scheduler.set_launch_callback(py_launch)

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

    cpdef enqueue_task(self, PyInnerTask task):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerTask* c_task = task.c_task
        c_self.enqueue_task(c_task)

    cpdef add_worker(self, PyInnerWorker worker):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerWorker* c_worker = worker.inner_worker
        c_self.add_worker(c_worker)

    cpdef enqueue_worker(self, PyInnerWorker worker):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerWorker* c_worker = worker.inner_worker
        c_self.enqueue_worker(c_worker)

    cpdef task_cleanup(self, PyInnerWorker worker, PyInnerTask task, int state):
        cdef InnerScheduler* c_self = self.inner_scheduler
        cdef InnerWorker* c_worker = worker.inner_worker
        cdef InnerTask* c_task = task.c_task
        with nogil:
            c_self.task_cleanup(c_worker, c_task, state)

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

    
class Resources:

    def __init__(self, vcus):
        self.resources = vcus
