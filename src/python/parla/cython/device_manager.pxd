import cython
cimport cython

from parla.cython.device cimport CUDADevice, CPUDevice, Device, CyDevice 

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device_manager.hpp" nogil:
    cdef cppclass DeviceManager:
        DeviceManager() except +
        void register_device(Device*) except +
        void print_registered_devices() except +
        int get_parray_id(long) except +


cdef class CyDeviceManager:
    cdef DeviceManager* cpp_device_manager_
    cpdef register_device(self, CyDevice cy_device)
    cpdef print_registered_devices(self)
    cpdef get_parray_id(self, global_device_id)
    cdef DeviceManager* get_cpp_device_manager(self)
