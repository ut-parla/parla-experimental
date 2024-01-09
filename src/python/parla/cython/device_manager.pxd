# cython: language_level=3
# cython: language=c++
import cython
cimport cython

from parla.cython.device cimport GPUDevice, CPUDevice, Device, CyDevice 

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device_manager.hpp" nogil:
    cdef cppclass DeviceManager:
        DeviceManager() except +
        void register_device(Device*) except +
        void print_registered_devices() except +
        int globalid_to_parrayid(int) except +
        int parrayid_to_globalid(int) except +


cdef class CyDeviceManager:
    cdef DeviceManager* cpp_device_manager_
    cpdef register_device(self, CyDevice cy_device)
    cpdef print_registered_devices(self)
    cpdef globalid_to_parrayid(self, global_dev_id)
    cpdef parrayid_to_globalid(self, parray_dev_id)
    cdef DeviceManager* get_cpp_device_manager(self)
