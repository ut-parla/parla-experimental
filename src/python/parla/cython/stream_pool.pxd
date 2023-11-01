

from libc.stdint cimport uintptr_t

cdef extern from "include/device_contexts.hpp":
    cdef cppclass InnerStreamPool:
        InnerStreamPool()
        void register_stream(int device_id, uintptr_t stream)
        uintptr_t get_stream(int device_id, int stream_id)


