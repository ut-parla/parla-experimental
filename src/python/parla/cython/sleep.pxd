#distutils: language = c++

from libc.stdint cimport intptr_t

cdef extern from "sleep.cu" nogil:
    cdef void gpu_sleeper(int dev, unsigned long t, intptr_t stream)
