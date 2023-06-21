import cython
cimport cython

from parla.cython.device_manager cimport DeviceManager
from parla.cython.device cimport Device, CyDevice
from parla.cython.cyparray cimport InnerPArray, CyPArray
from parla.cython.mm cimport LRUGlobalMemoryManager

from libc.stdint cimport uint32_t, uint64_t, int64_t
from libcpp  cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from libc.stdint cimport uintptr_t

cdef extern from "include/resources.hpp" nogil:
    cdef enum Resource:
        MEMORY = 0,
        VCUS = 1,

cdef extern from "include/gpu_utility.hpp" nogil:
    void cpu_busy_sleep(unsigned int microseconds)
    void gpu_busy_sleep(const int device, const unsigned long cycles,
                    uintptr_t stream_ptr)


cdef extern from "include/runtime.hpp" nogil:
    ctypedef void (*launchfunc_t)(void* py_scheduler, void* py_task, void* py_worker)
    ctypedef void (*stopfunc_t)(void* scheduler)

    void launch_task_callback(launchfunc_t func, void* py_scheduler, void* py_task, void* py_worker)
    void stop_callback(stopfunc_t func, void* scheduler)

    #ctypedef void* Ptr_t
    #ctypedef InnerTask* InnerTaskPtr_t

    cdef cppclass _StatusFlags "Task::StatusFlags":
        bool spawnable
        bool mappable
        bool reservable
        bool compute_runnable
        bool runnable

    cdef cppclass InnerTask:
        InnerTask()

        void set_scheduler(InnerScheduler* scheduler)
        void set_name(string name)
        void set_id(long long int i)
        void set_py_task(void *py_task)
        void set_priority(int p)

        void queue_dependency(InnerTask* task)
        _StatusFlags process_dependencies()
        void clear_dependencies()

        vector[void*] get_dependencies()
        vector[void*] get_dependents()
        vector[Device*]& get_assigned_devices() 
        void add_parray(InnerPArray* py_parray, int access_mode, int dev_id)
        bool notify_dependents_wrapper()

        void* get_py_task()

        int get_num_dependencies()
        int get_num_dependents()

        int get_num_blocking_dependencies()
        int get_num_unmapped_dependencies()

        string get_name()

        int set_state(int state)
        void add_device_req(void* dev_ptr, long mem_sz, int num_vcus)
        void begin_arch_req_addition()
        void end_arch_req_addition()
        void begin_multidev_req_addition()
        void end_multidev_req_addition()

        void add_event(uintptr_t event) except + 
        void add_stream(uintptr_t stream) except +

        void reset_events_streams() except +
        void handle_runahead_dependencies(int sync_type) except +
        void synchronize_events()   except +


    cdef cppclass InnerDataTask(InnerTask):
        void* get_py_parray()
        int get_access_mode()
        int get_device_id()

    cdef cppclass TaskBarrier:
        TaskBarrier() except +
        void add_tasks(vector[InnerTask*] tasks) except +
        void wait() except +
        void set_id(int64_t i) except +

    cdef cppclass InnerTaskSpace:
        InnerTaskSpace() except +
        void add_tasks(vector[int64_t] keys, vector[InnerTask*] tasks) except +
        void wait() except +
        void set_id(int64_t i) except +
        void get_tasks(vector[int64_t] keys, vector[InnerTask*] tasks) except +

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
        void get_task(InnerTask** task, bool* is_data_task)
        void remove_task() except +

        void wait()
        void stop()

    #ctypedef InnerWorker* InnerWorkerPtr_t

    cdef cppclass InnerScheduler:

        bool should_run
        
        InnerScheduler(LRUGlobalMemoryManager* cpp_memory_manager,  DeviceManager* cpp_device_manager)

        void set_num_workers(int num_workers)
        void set_py_scheduler(void* py_scheduler)
        void set_stop_callback(stopfunc_t func)

        bool get_should_run()

        void run() except +
        void stop()

        long long int get_memory_size_for_eviction(int dev_id) except +

        void activate_wrapper()

        void spawn_task(InnerTask* task)

        void add_worker(InnerWorker* worker)
        void enqueue_worker(InnerWorker* worker)
        void task_cleanup(InnerWorker* worker, InnerTask* task, int state) except +
        void task_cleanup_presync(InnerWorker* worker, InnerTask* task, int state) except +
        void task_cleanup_postsync(InnerWorker* worker, InnerTask* task, int state) except +

        int get_num_active_tasks()
        void increase_num_active_tasks()
        void decrease_num_active_tasks()

        #int get_num_active_workers()
        int get_num_running_tasks()
        int get_num_ready_tasks()
        int get_num_notified_workers()
        bool get_parray_state(uint32_t global_dev_idx, uint64_t parray_parent_id)

        void spawn_wait() except +

        void reserve_parray_to_tracker(InnerPArray* parray, int dev_id) except +
        void release_parray_from_tracker(InnerPArray* parray, int dev_id) except +

        void invoke_all_parrays_clear()
        bool get_clear_all_parrays()


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
