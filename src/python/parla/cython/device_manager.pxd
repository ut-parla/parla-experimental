import cython
cimport cython

from parla.cython.device cimport CUDADevice, CPUDevice, Device, CyDevice 

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device_manager.hpp" nogil:
    cdef cppclass DeviceManager:
        DeviceManager() except +
        void register_device(Device* device) except +
        #void register_cuda_device "register_device" (CUDADevice*)  except +
        #void register_cpu_device "register_device" (CPUDevice*) except +
        void print_registered_devices() except +


cdef class CyDeviceManager:
    cdef DeviceManager* cpp_device_manager_
    cpdef register_device(self, CyDevice cy_device)
    cpdef print_registered_devices(self)
    cdef DeviceManager* get_cpp_device_manager(self)
