from parla.cython.device_manager cimport DeviceManager

cdef extern from "include/memory_manager.hpp" nogil:
    cdef cppclass LRUGlobalMemoryManager:
        LRUGlobalMemoryManager(DeviceManager *)
        unsigned long long size(unsigned int device_id)
        void *remove_and_return_head_from_zrlist(unsigned int device_id)

cdef class CyMM:
    cdef LRUGlobalMemoryManager* _inner_mm
    cpdef size(self, int dev_id)
    cpdef remove_and_return_head_from_zrlist(self, int dev_id)
    cdef LRUGlobalMemoryManager* get_cpp_memory_manager(self)
