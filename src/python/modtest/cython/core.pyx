cdef class CythonMath:
    cpdef int add(self, int a, int b):
        return a + b

    cpdef int sub(self, int a, int b):
        return a - b


class CythonMathWrapper:

    def __init__(self):
        self.cython_math = CythonMath()

    def add(self, a, b):
        return self.cython_math.add(a, b)

    def sub(self, a, b):
        return self.cython_math.sub(a, b)

cdef class CppMathWrapper:
    cdef CppMath cpp_math

    def __cinit__(self):
        self.cpp_math = CppMath()
    
    def __init__(self):
        pass

    cpdef int add(self, int a, int b):
        return self.cpp_math.add(a, b)

    cpdef int sub(self, int a, int b):
        return self.cpp_math.sub(a, b)