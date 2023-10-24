#cython: language_level=3
#cython: language=c++
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
        void track_parray(InnerPArray&, uint32_t) except +
        void untrack_parray(InnerPArray&, uint32_t) except +
        void reserve_parray(InnerPArray&, Device*) except +
        void release_parray(InnerPArray&, Device*) except +
