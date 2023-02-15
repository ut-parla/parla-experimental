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

    cdef cppclass DeviceManager:
        DeviceManager() except +
        void RegisterDevice(Device*) except +
        void PrintRegisteredDevices() except +

    cdef cppclass DeviceSet:
        vector[void*] GetPyDevices() except +  
