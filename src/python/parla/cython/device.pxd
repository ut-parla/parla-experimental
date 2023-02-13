import cython
cimport cython

from libcpp.string cimport string

cdef extern from "include/device.hpp" nogil:
    cdef cppclass Device:
      int GetID()

    cdef cppclass CUDADevice(Device):
      CUDADevice(int) except +

    cdef cppclass CPUDevice(Device):
      CPUDevice(int) except +
