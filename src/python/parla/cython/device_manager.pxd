import cython
cimport cython

from parla.cython.device cimport Device, CyDevice 

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device_manager.hpp" nogil:
    cdef cppclass DeviceManager:
        DeviceManager() except +
        void RegisterDevice(Device*) except +
        void PrintRegisteredDevices() except +


cdef class CyDeviceManager:
    cdef DeviceManager* cpp_device_manager_
    cpdef register_device(self, CyDevice cy_device)
    cpdef print_registered_devices(self)
    cdef DeviceManager* get_cpp_device_manager(self)
