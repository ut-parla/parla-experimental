# distutils: language=c++

from libc.stdint cimport uint64_t

from .cyparray_state cimport PArrayState

#cdef extern from "parray.cpp":
#    pass

# a mapping between C++ PArray api to Cython PArray api
cdef extern from "include/parray.hpp" namespace "parray":
    cdef cppclass InnerPArray:
        InnerPArray() except +
        InnerPArray(void *, uint64_t, PArrayState *) except +
        void set_size(uint64_t)

cdef class CyPArray:
    # Hold a C++ instance which we're wrapping
    cdef InnerPArray* cpp_parray
    cdef InnerPArray* get_cpp_parray(self)
