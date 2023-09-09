import cython
cimport cython

from parla.cython.device cimport Device
from parla.cython.device_manager cimport DeviceManager
from parla.cython.cyparray cimport InnerPArray

from libc.stdint cimport uint32_t

"""
This module bridges Python and C++ PArray tracker.
This is implemented for moduel testing.
"""

cdef extern from "include/parray_tracker.hpp" nogil:
    cdef cppclass PArrayTracker:
        PArrayTracker(DeviceManager*) except +
