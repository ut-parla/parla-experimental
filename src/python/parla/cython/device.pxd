import cython
cimport cython

from libcpp.string cimport string

cdef extern from "include/device.hpp" nogil:
    cdef enum DevID:
      CUDA_GPU,
      CPU

    cdef cppclass Device:
      string GetName()
      DevID GetID()

    cdef cppclass CUDADevice(Device):
      CUDADevice(string, DevID) except +

    cdef cppclass CPUDevice(Device):
      CPUDevice(string, DevID) except +
