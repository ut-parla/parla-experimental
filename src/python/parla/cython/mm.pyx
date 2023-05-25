
from parla.cython import device_manager

from parla.cython.core cimport LRUGlobalMemoryManager
from parla.cython cimport device_manager
#from parla.cython.core import LRUGlobalMemoryManager

class PyMM:
    def __init__(self, dm: device_manager.PyDeviceManager):
        #print("PyMM constructor", flush=True)
        self._device_manager = device_manager
        self._cy_mm = CyMM(dm.get_cy_device_manager())
        #print("PyMM constructor [done]", flush=True)

    def size(self, dev_id: int):
        return self._cy_mm.size(dev_id)

    def remove_and_return_head_from_zrlist(self, dev_id: int):
        return self._cy_mm.remove_and_return_head_from_zrlist(dev_id)

    def get_cy_memory_manager(self):
        return self._cy_mm

    def print_memory_stats(self, device_id, label: str):
        import psutil
        import os
        print(f"[{label}] Memory tracking", flush=True)
        try:
            import cupy
            mempool = cupy.get_default_memory_pool()
            pinned_mempool = cupy.get_default_pinned_memory_pool()
            print((
                  f"\t GPU{device_id} {label} CuPy used bytes: {mempool.used_bytes()} \n"
                  f"\t GPU{device_id} {label} Free bytes: {mempool.free_bytes()} \n"
                  f"\t GPU{device_id} {label} Total bytes: {mempool.total_bytes()} \n"), flush=True)
        except ImportError:
            print("MM tracker only tracks CuPy memory for now.", flush=True)


cdef class CyMM:
#cdef LRUGlobalMemoryManager* _inner_mm

    def __cinit__(self, device_manager.CyDeviceManager cy_dm):
        print("CyMM cinit()", flush=True)
        self._inner_mm = new LRUGlobalMemoryManager(cy_dm.get_cpp_device_manager())
        print("CyMM cinit() [done]", flush=True)

    def __dealloc__(self):
        print("Deallocation CyMM\n")
        del self._inner_mm

    cpdef size(self, int dev_id):
        cdef LRUGlobalMemoryManager* c_self = self._inner_mm
        return c_self.size(dev_id)

    cpdef remove_and_return_head_from_zrlist(self, int dev_id):
        cdef LRUGlobalMemoryManager* c_self = self._inner_mm
        cdef void* py_parray = c_self.remove_and_return_head_from_zrlist(dev_id)
        if py_parray == NULL:
            return None
        else:
            return <object> py_parray

    cdef LRUGlobalMemoryManager* get_cpp_memory_manager(self):
        return self._inner_mm
