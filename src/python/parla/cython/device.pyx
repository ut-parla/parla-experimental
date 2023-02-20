################################################################################
# Cython implementations (Declarations are in device.pxd)
################################################################################

from parla.common.global_enums import DeviceType 

cdef class CyDevice:
    """
    A bridge between pure Python and C++ device objects.
    """
    def __dealloc__(self):
        del self._cpp_device

    cdef Device* get_cpp_device(self):
        return self._cpp_device


cdef class CyCUDADevice(CyDevice):
    """
    An inherited class from `CyDevice` for a device object specialized to CUDA.
    """
    def __cinit__(self, int dev_id, long mem_sz, long num_vucs, py_device):
        # This object will be deallocated at CyDevice's dealloc().
        self._cpp_device = new CUDADevice(dev_id, mem_sz, num_vucs, \
                                          <void *> py_device)

    def __init__(self, int dev_id, long mem_sz, long num_vcus, py_device):
        pass


cdef class CyCPUDevice(CyDevice):
    """
    An inherited class from `CyDevice` for a device object specialized to CPU.
    """
    def __cinit__(self, int dev_id, long mem_sz, long num_vucs, py_device):
        # This object will be deallocated at CyDevice's dealloc().
        self._cpp_device = new CPUDevice(dev_id, mem_sz, num_vucs, \
                                         <void *> py_device)

    def __init__(self, int dev_id, long mem_sz, long num_vcus, py_device):
        pass


################################################################################
# Python
################################################################################


class PyDevice:
    """
    This class is to abstract a single device in Python and manages
    a device context as a task runs in Python.
    """
    def __init__(self, dev_type, dev_type_name, dev_id: int):
        self._dev_type = dev_type
        self._device_name = dev_type_name + ":" + str(dev_id)

    def __enter__(self):
        print(f"Entered device, {self.get_name()}, context", flush=True)

    def __exit__(self):
        print(f"Exited device, {self.get_name()}, context", flush=True)

    def get_name(self):
        return self._device_name

    def get_cy_device(self):
        return self._cy_device

    def __repr__(self):
        return self._device_name


"""
Device instances in Python manage resource status.
TODO(hc): the device configuration will be packed in a data class soon.
"""

class PyCUDADevice(PyDevice):
    """
    An inherited class from `PyDevice` for a device object specialized to CUDA.
    """
    def __init__(self, dev_id: int, mem_sz: long, num_vucs: long):
        super().__init__(DeviceType.CUDA, "CUDA", dev_id)
        self._cy_device = CyCUDADevice(dev_id, mem_sz, num_vucs, self)


class PyCPUDevice(PyDevice):
    """
    An inherited class from `PyDevice` for a device object specialized to CPU.
    """
    def __init__(self, dev_id: int, mem_sz: long, num_vucs: long):
        super().__init__(DeviceType.CPU, "CPU", dev_id)
        self._cy_device = CyCPUDevice(dev_id, mem_sz, num_vucs, self)
