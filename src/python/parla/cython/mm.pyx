
from parla.cython import device_manager

from parla.cython.core cimport LRUGlobalEvictionManager
from parla.cython cimport device_manager
#from parla.cython.core import LRUGlobalEvictionManager

class PyMM:
    def __init__(self, dm: device_manager.PyDeviceManager):
        self._device_manager = device_manager
        self._cy_mm = CyMM(dm.get_cy_device_manager())

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
            print("MM tracker only supports CuPy memory status checking.", flush=True)


cdef class CyMM:

    def __cinit__(self, device_manager.CyDeviceManager cy_dm):
        self._inner_mm = new LRUGlobalEvictionManager(cy_dm.get_cpp_device_manager())

    def __dealloc__(self):
        del self._inner_mm

    cpdef size(self, int dev_id):
        cdef LRUGlobalEvictionManager* c_self = self._inner_mm
        return c_self.size(dev_id)

    cpdef remove_and_return_head_from_zrlist(self, int dev_id):
        cdef LRUGlobalEvictionManager* c_self = self._inner_mm
        cdef void* py_parray = c_self.remove_and_return_head_from_zrlist(dev_id)
        if py_parray == NULL:
            # TODO(hc): This path is actually not used.
            # It would be great if we can check if this python object is valid
            # at here; it can simplify our current mechanism a lot.
            return None
        else:
            return <object> py_parray

    cdef LRUGlobalEvictionManager* get_cpp_memory_manager(self):
        return self._inner_mm
