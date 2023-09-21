from parla.cython.resources cimport Resource

import cython
cimport cython

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device.hpp" nogil:

    cdef enum DeviceType:
        All "DeviceType::All"
        CPU "DeviceType::CPU"
        CUDA "DeviceType::CUDA"
        # TODO(hc): For now, we only support CUDA gpu devices.
        # Laster, it would be extended to more gpu types
        # like for AMD
        
    cdef cppclass Device:
        Device(string, int, long, long, void*)
        int get_id()
        int get_global_id()
        string get_name()
        long get_memory_size()
        long get_num_vcus()
        void *get_py_device()
        long long int query_resource(Resource)
        long long int query_reserved_resource(Resource)
        long long int query_mapped_resource(Resource)

    cdef cppclass CUDADevice(Device):
        CUDADevice(int, long, long, void*)

    cdef cppclass CPUDevice(Device):
        CPUDevice(int, long, long, void*)

    cdef cppclass DeviceSet:
        vector[void*] get_py_devices()


cdef class CyDevice:
    cdef Device* _cpp_device
    cdef Device* get_cpp_device(self)
    cpdef int get_global_id(self)
    cpdef long long int query_resource(self, int)
    cpdef long long int query_reserved_resource(self, int)
    cpdef long long int query_mapped_resource(self, int)
