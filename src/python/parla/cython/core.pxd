import cython
cimport cython

from libcpp  cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/runtime.hpp" nogil:

    ctypedef void (*launchfunc_t)(void* py_scheduler, void* py_task, void* py_worker)
    ctypedef void (*stopfunc_t)(void* scheduler)

    void launch_task_callback(launchfunc_t func, void* py_scheduler, void* py_task, void* py_worker)
    void stop_callback(stopfunc_t func, void* scheduler)

    #ctypedef void* Ptr_t
    #ctypedef InnerTask* InnerTaskPtr_t

    cdef cppclass InnerTask:
        InnerTask()
        void set_name(string name)
        void set_id(long long int i)
        void set_py_task(void *py_task)
        void set_priority(int p)
        void set_resources(string resource_name, float amount)

        void queue_dependency(InnerTask* task)
        bool process_dependencies()

        vector[void*] get_dependencies()
        vector[void*] get_dependents()
        void* get_py_task()

        int get_num_dependencies()
        int get_num_dependents()


        int get_num_blocking_dependencies()

        void set_state(int state)
        void set_complete(bool complete)
        int get_complete()


    #ctypedef InnerTask* InnerTaskPtr_t

    cdef cppclass InnerWorker:
        void* py_worker

        InnerWorker()
        InnerWorker(void* py_worker)

        void set_py_worker(void* py_worker)

    #ctypedef InnerWorker* InnerWorkerPtr_t

    cdef cppclass InnerScheduler:
        InnerScheduler()

        void set_num_workers(int num_workers)
        void set_resources(string resource_name, float amount)
        void set_py_scheduler(void* py_scheduler)
        void set_stop_callback(stopfunc_t func)
        void set_launch_callback(launchfunc_t func)

        void run()
        void stop()

        void activate_wrapper()

        void enqueue_task(InnerTask* task)
        void enqueue_tasks(vector[InnerTask*]& tasks)

        void add_worker(InnerWorker* worker)
        void enqueue_worker(InnerWorker* worker)
        void task_cleanup(InnerWorker* worker, InnerTask* task, int state)

        int get_num_active_tasks()
        void increase_num_active_tasks()
        void decrease_num_active_tasks()

        #int get_num_active_workers()
        int get_num_running_tasks()
        int get_num_ready_tasks()















