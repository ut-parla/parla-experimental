import cython
cimport cython

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device.hpp" nogil:
    cdef cppclass DeviceManager:
        DeviceManager() except +
        void RegisterCudaDevice(int, long, long, void*) except +
        void RegisterCpuDevice(int, long, long, void*) except +
        void PrintRegisteredDevices() except +

    cdef cppclass DeviceSet:
        vector[void*] GetPyDevices() except +  
