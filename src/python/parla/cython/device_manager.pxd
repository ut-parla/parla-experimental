import cython
cimport cython

from parla.cython.device cimport CUDADevice, CPUDevice, ParlaDevice, CyDevice 

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "include/device_manager.hpp" nogil:

    int g2p "globalid_to_parrayid" (int) 
    unsigned int p2g "parrayid_to_globalid" (int)

    cdef cppclass DeviceManager:
        DeviceManager() 
        void register_device(ParlaDevice*) 
        void print_registered_devices() 
        void free_memory_by_parray_id(int parray_dev_id, unsigned long memory_size)
        void free_memory(unsigned int global_dev_id, unsigned long memory_size)
        

cdef class CyDeviceManager:
    cdef DeviceManager* cpp_device_manager_
    cpdef register_device(self, CyDevice cy_device)
    cpdef print_registered_devices(self)
    cpdef globalid_to_parrayid(self, global_dev_id)
    cpdef parrayid_to_globalid(self, parray_dev_id)
    cdef DeviceManager* get_cpp_device_manager(self)
    cpdef free_memory(self, parray_dev_id, memory_size)
