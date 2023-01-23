cdef class CythonMath:
    cpdef int add(self, int a, int b)
    cpdef int sub(self, int a, int b)

cdef extern from "backend.hpp" nogil:
    cdef cppclass CppMath:
        CppMath()
        int add(int a, int b)
        int sub(int a, int b)

