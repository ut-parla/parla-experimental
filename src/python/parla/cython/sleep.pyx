from libc.stdint cimport intptr_t
import time 

def gpu_sleep(dev, t, stream):
    cdef int c_dev = dev
    cdef unsigned long c_t = t
    cdef intptr_t c_stream = stream.ptr
    with nogil:
        gpu_sleeper(c_dev, c_t, c_stream)
