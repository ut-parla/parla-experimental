#cython: language_level=3
#cython: language=c++
from ..cython.resources cimport Resource

import cython
cimport cython

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device.hpp" nogil:

    cdef enum DeviceType:
        All "DeviceType::All"
        CPU "DeviceType::CPU"
        GPU "DeviceType::GPU"
        
    cdef cppclass Device:
        Device(string, int, long, long, void*) except +
        int get_id() except +
        int get_global_id() except +
        string get_name() except +
        long get_memory_size() except +
        long get_num_vcus() except +
        void *get_py_device() except +

    cdef cppclass GPUDevice(Device):
        GPUDevice(int, long, long, void*) except +

    cdef cppclass CPUDevice(Device):
        CPUDevice(int, long, long, void*) except +

    cdef cppclass DeviceSet:
        vector[void*] get_py_devices() except +  


cdef class CyDevice:
    cdef Device* _cpp_device
    cdef Device* get_cpp_device(self)
    cpdef int get_global_id(self)
