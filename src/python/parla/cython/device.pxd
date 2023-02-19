import cython
cimport cython

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device.hpp" nogil:
    cdef cppclass Device:
        Device(string, int, long, long, void*) except +
        int GetID() except +
        string GetName() except +
        long GetMemorySize() except +
        long GetNumVCUs() except +

    cdef cppclass CUDADevice(Device):
        CUDADevice(int, long, long, void*) except +

    cdef cppclass CPUDevice(Device):
        CPUDevice(int, long, long, void*) except +

    cdef cppclass DeviceSet:
        vector[void*] GetPyDevices() except +  


cdef class CyDevice:
    cdef Device* _cpp_device
    cdef Device* get_cpp_device(self)
