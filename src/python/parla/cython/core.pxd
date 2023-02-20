import cython
cimport cython

from parla.cython.device_manager cimport DeviceManager

from libcpp  cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/runtime.hpp" nogil:

    ctypedef void (*launchfunc_t)(void* py_scheduler, void* py_task, void* py_worker)
    ctypedef void (*stopfunc_t)(void* scheduler)

    void cpu_busy_sleep(unsigned int microseconds)

    void launch_task_callback(launchfunc_t func, void* py_scheduler, void* py_task, void* py_worker)
    void stop_callback(stopfunc_t func, void* scheduler)

    #ctypedef void* Ptr_t
    #ctypedef InnerTask* InnerTaskPtr_t

    cdef cppclass InnerTask:
        InnerTask()

        void set_scheduler(InnerScheduler* scheduler)
        void set_name(string name)
        void set_id(long long int i)
        void set_py_task(void *py_task)
        void set_priority(int p)
        void set_resources(string resource_name, float amount)

        void queue_dependency(InnerTask* task)
        bool process_dependencies()
        void clear_dependencies()

        vector[void*] get_dependencies()
        vector[void*] get_dependents()
        bool notify_dependents_wrapper()

        void* get_py_task()

        int get_num_dependencies()
        int get_num_dependents()


        int get_num_blocking_dependencies()

        void set_state(int state)
        void set_complete()
        int get_complete()


    #ctypedef InnerTask* InnerTaskPtr_t

    cdef cppclass InnerWorker:
        void* py_worker
        InnerTask* task

        bool ready

        InnerWorker()
        InnerWorker(void* py_worker)

        void set_py_worker(void* py_worker)
        void set_scheduler(InnerScheduler* scheduler
        )
        void set_thread_idx(int idx)
        void assign_task(InnerTask* task)
        InnerTask* get_task()
        void remove_task() except +

        void wait()
        void stop()

    #ctypedef InnerWorker* InnerWorkerPtr_t

    cdef cppclass InnerScheduler:

        bool should_run
        
        InnerScheduler(DeviceManager* cpp_device_manager)

        void set_num_workers(int num_workers)
        void set_resources(string resource_name, float amount)
        void set_py_scheduler(void* py_scheduler)
        void set_stop_callback(stopfunc_t func)
        void set_launch_callback(launchfunc_t func)

        void run() except +
        void stop()

        void activate_wrapper()

        void spawn_task(InnerTask* task, bool should_enqueue)
        void enqueue_task(InnerTask* task)
        void enqueue_tasks(vector[InnerTask*]& tasks)

        void add_worker(InnerWorker* worker)
        void enqueue_worker(InnerWorker* worker)
        void task_cleanup(InnerWorker* worker, InnerTask* task, int state) except +

        int get_num_active_tasks()
        void increase_num_active_tasks()
        void decrease_num_active_tasks()

        #int get_num_active_workers()
        int get_num_running_tasks()
        int get_num_ready_tasks()
        int get_num_notified_workers()

        void spawn_wait() except +




cdef extern from "include/profiling.hpp" nogil:
    void initialize_log(string filename)
    void write_log(string filename)

    void log_task_msg(int t, string msg)
    void log_worker_msg(int t, string msg)
    void log_scheduler_msg(int t, string msg)

    void log_task_1[T](int t, string msg, T* obj)
    void log_worker_1[T](int t, string msg, T* obj)
    void log_scheduler_1[T](int t, string msg, T* obj)


    void log_task_2[T, G](int t, string msg1, T* obj, string msg2, G* obj2)
    void log_worker_2[T, G](int t, string msg1, T* obj, string msg2, G* obj2)
    void log_scheduler_2[T, G](int t, string msg1, T* obj, string msg2, G* obj2)





