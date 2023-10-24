#cython: language_level=3
#cython: language=c++
"""!
@file cyparray.pyx
@brief Contains the core intermediate cython wrapper classes for PArray.
"""

from .cyparray cimport InnerPArray
from .cyparray_state cimport CyPArrayState
from ..common.parray.core import PArray

# a Cython wrapper class around C++ PArray
cdef class CyPArray:

    def __init__(self, py_parray, uint64_t id, uint64_t parent_id, py_parent_parray, \
                 CyPArrayState parray_state, int num_devices):
        pass

    def __cinit__(self, py_parray, uint64_t id, uint64_t parent_id, py_parent_parray, \
                  CyPArrayState parray_state, int num_devices):
        cdef InnerPArray* cpp_parent_parray = NULL
        cdef CyPArray cy_parent_parray
        if py_parent_parray is not None and py_parent_parray.ID != py_parray.ID:
            cy_parent_parray = py_parent_parray.cy_parray
            cpp_parent_parray = cy_parent_parray.get_cpp_parray()

        self.cpp_parray = new InnerPArray(\
            <void *> py_parray, id, parent_id, cpp_parent_parray,
            parray_state.get_cpp_parray_state(), num_devices)

    def __dealloc__(self):
        del self.cpp_parray

    def set_size(self, new_size):
        self.cpp_parray.set_size(new_size)

    cdef InnerPArray* get_cpp_parray(self):
        return self.cpp_parray

    cpdef get_num_active_tasks(self, int global_dev_id):
        return self.cpp_parray.get_num_active_tasks(global_dev_id)

    cpdef get_parray_parentid(self):
        return self.cpp_parray.get_parray_parentid()
