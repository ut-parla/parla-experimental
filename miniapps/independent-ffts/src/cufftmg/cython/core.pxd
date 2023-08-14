from numpy cimport uint64_t 

cdef extern from "cufftmg.hpp":
    cdef cppclass FFTHandler:
        FFTHandler()
        void configure(int*, int, int, uint64_t*, uint64_t*)
        void set_workspace(uint64_t* workspace)
        void execute(void**, void**, int)
        void empty()
