#cython: language_level=3
#cython: language=c++
import cython
cimport cython

cdef extern from "include/resources.hpp" nogil:
    cdef enum Resource:
        Memory "Resource:Memory"
        VCU "Resource::VCU"
        MAX "Resource::MAX"
