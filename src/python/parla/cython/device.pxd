from parla.cython.resources cimport Resource

import cython
cimport cython

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device.hpp" nogil:

    cdef enum ParlaDeviceType:
        All "ParlaDeviceType::All"
        CPU "ParlaDeviceType::CPU"
        CUDA "ParlaDeviceType::CUDA"
        # TODO(hc): For now, we only support CUDA gpu devices.
        # Laster, it would be extended to more gpu types
        # like for AMD
        
    cdef cppclass ParlaDevice:
        ParlaDevice(string, int, long, long, void*) except +
        int get_id() except +
        int get_global_id() except +
        string get_name() except +
        long get_memory_size() except +
        long get_num_vcus() except +
        void *get_py_device() except +
        long long int query_resource(Resource) except +
        long long int query_reserved_resource(Resource) except +
        long long int query_mapped_resource(Resource) except +

    cdef cppclass CUDADevice(ParlaDevice):
        CUDADevice(int, long, long, void*) except +

    cdef cppclass CPUDevice(ParlaDevice):
        CPUDevice(int, long, long, void*) except +

    cdef cppclass DeviceSet:
        vector[void*] get_py_devices() except +  


cdef class CyDevice:
    cdef ParlaDevice* _cpp_device
    cdef ParlaDevice* get_cpp_device(self)
    cpdef int get_global_id(self)
    cpdef long long int query_resource(self, int)
    cpdef long long int query_reserved_resource(self, int)
    cpdef long long int query_mapped_resource(self, int)
