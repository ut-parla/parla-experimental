from parla.common.global_dataclasses import DeviceConfig 
from parla.cython.device import CyDevice, PyCUDADevice, PyCPUDevice
from parla.cython.device cimport Device

import nvidia_smi

def get_cuda_device_info():
    nvidia_smi.nvmlInit()
    num_of_gpus = nvidia_smi.nvmlDeviceGetCount()
    print("Number of GPUs:", num_of_gpus)
    if num_of_gpus > 0:
        for i in range(num_of_gpus):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU-{i}: GPU-Memory: {mem_info.used}/{mem_info.total} Bytes")
            dev_name = nvidia_smi.nvmlDeviceGetName(handle).decode("utf-8")
            print(f"Device name: {dev_name}")


get_cuda_device_info()


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
