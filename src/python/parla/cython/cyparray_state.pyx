# cython: language_level=3
# cython: language=c++
"""!
@file cyparray_state.pyx
@brief Contains the core intermediate cython wrapper classes for PArray Coherence.
"""

# a Cython wrapper class around C++ PArrayState
cdef class CyPArrayState:
    def __cinit__(self):
        self.cpp_parray_state = new PArrayState()

    def __init__(self):
        pass

    def __dealloc__(self):
        del self.cpp_parray_state

    def set_exist_on_device(self, device_id, exist):
        self.cpp_parray_state.set_exist_on_device(device_id, exist)

    def set_valid_on_device(self, device_id, valid):
        self.cpp_parray_state.set_valid_on_device(device_id, valid)
    
    cdef PArrayState* get_cpp_parray_state(self):
        return self.cpp_parray_state
