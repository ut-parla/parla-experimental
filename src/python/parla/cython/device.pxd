import cython
cimport cython

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device.hpp" nogil:
    cdef cppclass Device:
        int GetID()
        string GetName()

    cdef cppclass CUDADevice(Device):
        CUDADevice(int) except +

    cdef cppclass CPUDevice(Device):
        CPUDevice(int) except +

    cdef cppclass DeviceManager:
        DeviceManager(int, int) except +
        void RegisterDevices() except +
        vector[Device]& GetAllDevices() except +
