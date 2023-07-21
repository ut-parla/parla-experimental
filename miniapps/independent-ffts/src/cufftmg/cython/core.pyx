cimport numpy as np
import numpy as np
import cupy as cp
import cython
import crosspy
import cupy

from numpy cimport uint64_t 
from parla.tasks import get_current_context

def import_test():
    print("IMPORT SUCCESS", flush=True)


cdef class CyFFTHandler:
    cdef FFTHandler* handler
    cdef int n_devices

    def __cinit__(self):
        self.handler = new FFTHandler()
    
    def __init__(self):
        pass

    def configure(self, size=0, device_ids=[], streams=[]):

        cdef int n_devices = len(device_ids)
        self.n_devices = n_devices

        cdef int c_size = size

        cdef FFTHandler* fft_handler = self.handler

        cdef int[:] c_device_ids = np.empty(n_devices, dtype=np.int32)

        cdef uint64_t[:] c_streams = np.empty(n_devices, dtype=np.uint64)

        cdef uint64_t[:] c_work_sizes = np.zeros(n_devices, dtype=np.uint64)

        for i in range(n_devices):
            c_device_ids[i] = device_ids[i]
            stream = streams[i]
            c_streams[i] = <uint64_t> streams[i].ptr

        fft_handler.configure(<int*> &c_device_ids[0], c_size, n_devices, <uint64_t*> &c_streams[0], <uint64_t*> &c_work_sizes[0])
        print("We didn't crash!", flush=True)

    def fft2(self, array):
        context = get_current_context()
        has_context = True
        print("Context: ", context, flush=True)

        if context is None:
            has_context = False

        cdef uint64_t[:] c_array = np.empty(self.n_devices, dtype=np.uint64)

        cdef FFTHandler* fft_handler = self.handler

        for i, local_block in enumerate(array.block_view()):
            if has_context:
                print("Using Context to Serialize", flush=True)
                with context.devices[i] as device:
                    c_array[i] = <uint64_t> local_block[0].data.mem.ptr
                print("Finished Using Context to Serialize", flush=True)
            else:
                c_array[i] = <uint64_t> local_block[0].data.mem.ptr

        fft_handler.execute(<void**> &c_array[0], <void**> &c_array[0], -1)

    def ifft2(self, array):
        context = get_current_context()

        has_context = True
        if context is None:
            has_context = False

        cdef uint64_t[:] c_array = np.empty(self.n_devices, dtype=np.uint64)
        cdef FFTHandler* fft_handler = self.handler

        for i, local_block in enumerate(array.block_view()):
            if has_context:
                with context.devices[i] as device:
                    c_array[i] = <uint64_t> local_block[0].data.mem.ptr
            else:
                c_array[i] = <uint64_t> local_block[0].data.mem.ptr

        fft_handler.execute(<void**> &c_array[0], <void**> &c_array[0], 1)

    def __dealloc__(self):
        print("Deleting FFT handler", flush=True)
        del self.handler


class Handler:
    def __init__(self):
        self.handler = CyFFTHandler()

    def configure(self, size=0, device_ids=[], streams=[]):
        self.handler.configure(size, device_ids, streams)

    def fft2(self, array):
        self.handler.fft2(array)

    def ifft2(self, array):
        self.handler.ifft2(array)







    
