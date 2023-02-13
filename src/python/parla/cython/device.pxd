import cython
cimport cython

from libcpp.string cimport string

cdef extern from "include/device.hpp" nogil:
    cdef enum ArchID:
      CUDA_GPU,
      CPU

    cdef cppclass Architecture:
      Architecture(string, ArchID) except + 
      string GetName()
      ArchID GetID()

    cdef cppclass CUDAArch(Architecture):
      CUDAArch()

    cdef cppclass CPUArch(Architecture):
      CPUArch()

    cdef cppclass Device:
      pass

    cdef cppclass CUDADevice(Device):
      pass

    cdef cppclass CPUDevice(Device):
      pass
