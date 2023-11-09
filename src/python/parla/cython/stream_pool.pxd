

from libc.stdint cimport uintptr_t

cdef extern from "include/device_contexts.hpp":
    cdef cppclass InnerStreamPool:
        InnerStreamPool()
        void push_stream(int device_id, uintptr_t stream)
        uintptr_t pop_stream(int device_id)

        void push_event(int device_id, uintptr_t event)
        uintptr_t pop_event(int device_id)

        

cdef class CyStreamPool:
    cdef InnerStreamPool* _c_pool
    cdef dict _pool
    cdef int _per_device
    cdef list _device_list

