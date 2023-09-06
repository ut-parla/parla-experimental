# distutils: language=c++

from libc.stdint cimport uint32_t, uint64_t

from .cyparray_state cimport PArrayState

#cdef extern from "parray.cpp":
#    pass

# a mapping between C++ PArray api to Cython PArray api
cdef extern from "include/parray.hpp" namespace "parray":
    cdef cppclass InnerPArray:
        InnerPArray(void *, uint64_t, uint64_t, InnerPArray *, PArrayState *i, uint32_t) except +
        void set_size(uint64_t)
        uint64_t get_num_active_tasks(uint32_t global_dev_id) except +
        const uint64_t get_parent_id() except +

cdef class CyPArray:
    # Hold a C++ instance which we're wrapping
    cdef InnerPArray* cpp_parray
    cdef InnerPArray* get_cpp_parray(self)
    cpdef get_num_active_tasks(self, int global_dev_id)
    cpdef get_parray_parentid(self)
