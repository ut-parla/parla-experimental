# distutils: language=c++

from .cyparray cimport InnerPArray
from .cyparray_state cimport CyPArrayState

# a Cython wrapper class around C++ PArray
cdef class CyPArray:

    def __init__(self, py_parray, uint64_t id, CyPArrayState parray_state):
        pass

    def __cinit__(self, py_parray, uint64_t id, CyPArrayState parray_state):
        self.cpp_parray = new InnerPArray(<void *> py_parray, id, parray_state.get_cpp_parray_state())

    def __dealloc__(self):
        del self.cpp_parray

    def set_size(self, new_size):
        self.cpp_parray.set_size(new_size)

    cdef InnerPArray* get_cpp_parray(self):
        return self.cpp_parray
