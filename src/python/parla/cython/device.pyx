################################################################################
# Cython implementations (Declarations are in device.pxd)
################################################################################


cdef class CyDeviceManager:
    """
    A bridge between pure Python and C++ device managers.
    """
    def __cinit__(self):
        self.cpp_device_manager_ = new DeviceManager()

    def __init__(self):
        pass

    def __dealloc__(self):
        del self.cpp_device_manager_

    cpdef register_device(self, CyDevice cy_device):
        """ Register devices to the c++ runtime. """
        cdef Device* cpp_device = cy_device.get_cpp_device()
        self.cpp_device_manager_.RegisterDevice(cpp_device)

    cpdef print_registered_devices(self):
        self.cpp_device_manager_.PrintRegisteredDevices()

    cdef DeviceManager* get_cpp_device_manager(self):
        return self.cpp_device_manager_


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
    def __init__(self, dev_type_name, dev_id: int):
        self._device_name = dev_type_name + ":" + str(dev_id)
        print(f"Device: {self._device_name} is registered.", flush=True)

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
        super().__init__("CUDA", dev_id)
        self._cy_device = CyCUDADevice(dev_id, mem_sz, num_vucs, self)


class PyCPUDevice(PyDevice):
    """
    An inherited class from `PyDevice` for a device object specialized to CPU.
    """
    def __init__(self, dev_id: int, mem_sz: long, num_vucs: long):
        super().__init__("CPU", dev_id)
        self._cy_device = CyCPUDevice(dev_id, mem_sz, num_vucs, self)


class PyDeviceManager:
    """
    A device manager manages device objects and provides their information.
    The Parla runtime should access device information through a device manager.
    A single device manager for each Python and C++ is created and held
    by each schedulers.
    """

    def __init__(self, config = None):
        self.cy_device_manager = CyDeviceManager()
        self.py_registered_devices = []
        if config == None:
            # For now, use hand-written device specs and register devices.
            # TODO(hc): query to OS or get configuration from users.
            # TODO(hc): pack those config. to a data class.
            self.register_cuda_device(0, 16000000000, 1)
            self.register_cuda_device(1, 16000000000, 1)
            self.register_cuda_device(2, 16000000000, 1)
            self.register_cuda_device(3, 16000000000, 1)
            self.register_cpu_device(0, 190000000000, 1)
 
    def register_cuda_device(self, dev_id: int, mem_sz: int, num_vcus: int):
        """
        Register a CUDA device to the both Python/C++ runtime.
        """
        py_cuda_device = PyCUDADevice(dev_id, mem_sz, num_vcus)
        self.py_registered_devices.append(py_cuda_device)
        cy_cuda_device = py_cuda_device.get_cy_device()
        self.cy_device_manager.register_device(cy_cuda_device)

    def register_cpu_device(self, dev_id: int, mem_sz: int, num_vcus: int):
        """
        Register a CPU device to the both Python/C++ runtime.
        """
        py_cpu_device = PyCPUDevice(dev_id, mem_sz, num_vcus)
        self.py_registered_devices.append(py_cpu_device)
        cy_cpu_device = py_cpu_device.get_cy_device()
        self.cy_device_manager.register_device(cy_cpu_device)

    def print_registered_devices(self):
        for dev in self.py_registered_devices:
            print(f"Registered device: {dev}", flush=True)
        self.cy_device_manager.print_registered_devices()

    def get_cy_device_manager(self):
        return self.cy_device_manager

