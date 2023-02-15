from parla.cython import core

cdef class CyDeviceManager:
    """
    This class manages devices on the current system.
    For convenience, it registers devices specified by users to both
    pure Python and Cython side; So, it replicates processes.
    """
    cdef DeviceManager* cpp_device_manager_

    def __cinit__(self):
        self.cpp_device_manager_ = new DeviceManager()

    def __init__(self):
        pass

    def __dealloc__(self):
        del self.cpp_device_manager_

    cpdef register_cuda_device(self, dev_id, memory, vcu, py_task):
        self.cpp_device_manager_.RegisterCudaDevice(dev_id, memory, vcu, <void *> py_task)

    cpdef register_cpu_device(self, dev_id, memory, vcu, py_task):
        self.cpp_device_manager_.RegisterCpuDevice(dev_id, memory, vcu, <void *> py_task)

    cpdef print_registered_devices(self):
        self.cpp_device_manager_.PrintRegisteredDevices()



class PyDevice:

    def __init__(self, dev_type_name, dev_id: int):
        self._device_name = dev_type_name + ":" + str(dev_id)
        print(f"Device: {self._device_name} is registered.", flush=True)

    def __enter__(self):
        print(f"Entered device, {self.get_name()}, context", flush=True)

    def __exit__(self):
        print(f"Exited device, {self.get_name()}, context", flush=True)

    def get_name(self):
        return self._device_name

    def __repr__(self):
        return self._device_name


"""
Device instances in Python manage resource status.
"""

class PyCUDADevice(PyDevice):

    def __init__(self, dev_id: int):
        super().__init__("CUDA", dev_id)


class PyCPUDevice(PyDevice):

    def __init__(self, dev_id: int):
        super().__init__("CPU", dev_id)


class PyDeviceManager:

    def __init__(self, config = None):
        self.cy_device_manager = CyDeviceManager()
        self.py_registered_devices = []
        if config == None:
            # For now, use hand-written device specs and register devices.
            # TODO(hc): query to OS or get configuration from users.
            self.register_cuda_device(0, 16000000000, 1)
            self.register_cuda_device(1, 16000000000, 1)
            self.register_cuda_device(2, 16000000000, 1)
            self.register_cuda_device(3, 16000000000, 1)
            self.register_cpu_device(0, 190000000000, 1)
 
    def register_cuda_device(self, dev_id: int, memory: int, vcu: int):
        py_cuda_device = PyCUDADevice(dev_id)
        self.py_registered_devices.append(py_cuda_device)
        self.cy_device_manager.register_cuda_device(\
                               dev_id, memory, vcu, py_cuda_device)

    def register_cpu_device(self, dev_id: int, memory: int, vcu: int):
        py_cpu_device = PyCPUDevice(dev_id)
        self.py_registered_devices.append(py_cpu_device)
        self.cy_device_manager.register_cpu_device(\
                               dev_id, memory, vcu, py_cpu_device)

    def print_registered_devices(self):
        for dev in self.py_registered_devices:
            print(f"Registered device: {dev}", flush=True)
        self.cy_device_manager.print_registered_devices()

