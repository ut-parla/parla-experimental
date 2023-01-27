cdef void callback_launch(void* python_scheduler, void* python_task, void*
        python_worker) nogil:
    with gil:
        #print("Inside callback to cython", flush=True)
        task = <object>python_task
        scheduler = <object>python_scheduler
        worker = <object>python_worker

        scheduler.cpp_callback(task, worker)
        #print("Done with callback", flush=True)
        #(<object>python_function)(<object>python_input)

cdef void callback_stop(void* python_function) nogil:
    with gil:
        #print("Inside callback to cython (stop)", flush=True)
        scheduler = <object>python_function
        scheduler.stop()

        #(<object>python_function)(<object>python_input)



cdef class PyInnerTask:
    cdef InnerTask* c_task

    def __cinit__(self):
        cdef InnerTask* _c_task
        _c_task = new InnerTask()
        self.c_task = _c_task

    def __init__(self, long long int idx, object python_task, float vcus):
        cdef InnerTask* _c_task
        _c_task = self.c_task

        #name = python_task._taskid.full_name
        name = "test"
        name = name.encode('utf-8')
        _c_task.set_name(name)

        _c_task.set_id(idx)
        _c_task.set_py_task(<void *> python_task)

        priority = 0
        _c_task.set_priority(priority)

        resource_type = "vcus"
        resource_type = resource_type.encode('utf-8')
        _c_task.set_resources(resource_type, vcus)

    def __dealloc__(self):
        del self.c_task

    cpdef get_py_task(self):
        cdef InnerTask* c_self = self.c_task
        return <object> c_self.get_py_task()

    cpdef add_dependencies(self, dependency_list):
        cdef InnerTask* c_self = self.c_task

        cdef PyInnerTask dependency
        cdef InnerTask* c_dependency

        for d in dependency_list:
            dependency = d.inner_task
            c_dependency = dependency.c_task
            c_self.queue_dependency(c_dependency)


        #TODO: Remove GIL here?
        status = c_self.process_dependencies()
        return status

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

    def __dealloc__(self):
        del self.inner_scheduler

    cpdef run(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        c_self.run()

    cpdef stop(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        c_self.stop()

    cpdef activate_wrapper(self):
        cdef InnerScheduler* c_self = self.inner_scheduler
        c_self.activate_wrapper()

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

    

    
class Task:

    def __init__(self, name="Default", dependencies=None):

        self.name = name

        self.vcus = 1
        self.id = id(self)
        self.inner_task = PyInnerTask(self.id, self, self.vcus)

        if dependencies is not None:
            self.add_dependencies(dependencies)

    def add_dependencies(self, dependency_list):
        self.inner_task.add_dependencies(dependency_list)

    def get_num_dependencies(self):
        return self.inner_task.get_num_dependencies()

    def get_num_dependents(self):
        return self.inner_task.get_num_dependents()

    def get_num_blocking_dependencies(self):
        return self.inner_task.get_num_blocking_dependencies()

    def get_dependencies(self):
        dependency_list = self.inner_task.get_dependencies()
        return dependency_list

    def get_dependents(self):
        dependent_list = self.inner_task.get_dependents()
        return dependent_list

    def set_state(self, state):
        self.inner_task.set_state(state)

    def get_state(self):
        return self.inner_task.get_state()

    def set_complete(self):
        self.inner_task.set_complete()

    




