from parla.cython import device
from parla.cython.device cimport Device
from parla.common.global_enums import DeviceType

try:
    import cupy
except ImportError:
    cupy = None

from typing import FrozenSet, Collection, Iterable, Set, Tuple, List

import os
import psutil
import yaml

VCU_BASELINE=1000

PyDevice = device.PyDevice
PyCUDADevice = device.PyCUDADevice
PyCPUDevice = device.PyCPUDevice
PyArchitecture = device.PyArchitecture
PyCUDAArchitecture = device.PyCUDAArchitecture
PyCPUArchitecture = device.PyCPUArchitecture
DeviceResource = device.DeviceResource
DeviceResourceRequirement = device.DeviceResourceRequirement

# Architecture declaration.
# To use these in the placement of @spawn,
# declare these as global variables.
cuda = PyCUDAArchitecture(DeviceType.CUDA)
cpu = PyCPUArchitecture(DeviceType.CPU)

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


class PrintableFrozenSet(frozenset):
    """
    Add __repr__ to frozenset.
    """
    def get_name(self):
        # TODO(hc): better way?
        name = "[FrozenSet] "
        for elem in self:
            name += str(elem) + ","
        return name[:-1]

    def __repr__(self):
        return self.get_name()


class PyDeviceManager:
    """
    A device manager manages device objects and provides their information.
    The Parla runtime should access device information through a device manager.
    A single device manager for each Python and C++ is created and held
    by each schedulers.
    """

    def __init__(self, dev_config = None):
        self.cy_device_manager = CyDeviceManager()
        # TODO(hc): I don't know a better way to register architecture.
        self.py_registered_archs = [None] * 2
        self.py_registered_archs[DeviceType.CPU] = cpu
        self.py_registered_archs[DeviceType.CUDA] = cuda
        # TODO(hc): pack those config. to a data class.
        if dev_config == None or dev_config == "":
            self.register_cuda_devices_cupy()
            self.register_cpu_devices()
        else:
            self.parse_config_and_register_devices(dev_config)
        self.register_devices_to_cpp()

    def register_cuda_devices_cupy(self):
        if cupy is not None:
            try:
                num_of_gpus = cupy.cuda.runtime.getDeviceCount()
            except cupy.cuda.runtime.CUDARuntimeError:
                num_of_gpus = 0
        else:
            num_of_gpus = 0

        if num_of_gpus > 0:
            py_cuda_arch = self.py_registered_archs[DeviceType.CUDA]
            for dev_id in range(num_of_gpus):
                gpu_dev = cupy.cuda.Device(dev_id)
                mem_info = gpu_dev.mem_info # tuple of free and total memory
                                            # in bytes.
                mem_sz = long(mem_info[1])
                py_cuda_device = PyCUDADevice(dev_id, mem_sz, VCU_BASELINE)
                py_cuda_arch.add_device(py_cuda_device)

    def register_cpu_devices(self):
        # Get the number of usable CPUs from this process.
        # This might not be equal to the number of CPUs in the system.
        num_cores = len(os.sched_getaffinity(0))
        mem_sz = long(psutil.virtual_memory().total)
        py_cpu_device = PyCPUDevice(0, mem_sz, num_cores * VCU_BASELINE)
        py_cpu_arch = self.py_registered_archs[DeviceType.CPU]
        py_cpu_arch.add_device(py_cpu_device)

    def register_devices_to_cpp(self):
        """
        Register devices to the both Python/C++ runtime.
        """
        for py_arch in self.py_registered_archs:
            for py_device in py_arch.devices:
                cy_device = py_device.get_cy_device()
                self.cy_device_manager.register_device(cy_device)

    def print_registered_devices(self):
        print("Python devices:", flush=True)
        for dev in self.py_registered_archs:
            print(f"\t Registered device: {dev}", flush=True)
        self.cy_device_manager.print_registered_devices()

    def get_cy_device_manager(self):
        return self.cy_device_manager

    def parse_config_and_register_devices(self, yaml_config):
        with open(yaml_config, "r") as f:
            parsed_configs = yaml.safe_load(f)
            # Parse CPU device information.
            cpu_num_cores = parsed_configs["CPU"]["num_cores"]
            if cpu_num_cores > 0:
                cpu_mem_sz = parsed_configs["CPU"]["mem_sz"]
                py_cpu_device = PyCPUDevice(0, cpu_mem_sz, \
                                            cpu_num_cores * VCU_BASELINE) 
                py_cpu_arch = self.py_registered_archs[DeviceType.CPU]
                py_cpu_arch.add_device(py_cpu_device)
            gpu_num_devices = parsed_configs["GPU"]["num_devices"]
            if gpu_num_devices > 0:
                gpu_mem_sizes = parsed_configs["GPU"]["mem_sz"]
                assert(gpu_num_devices == len(gpu_mem_sizes)) 
                py_cuda_arch = self.py_registered_archs[DeviceType.CUDA]
                for dev_id in range(gpu_num_devices):
                    py_cuda_device = PyCUDADevice(dev_id, \
                                                  gpu_mem_sizes[dev_id], \
                                                  VCU_BASELINE)
                    py_cuda_arch.add_device(py_cuda_device)

    def get_all_devices(self):
        devs = []
        for arch in self.py_registered_archs:
            for dev in arch.devices:
                devs.append(dev)
        return devs

    def unpack_devices(self, placement_component):
        """ Unpack the placement and return a list of
            devices. The placement can be an iteratable collection
            and this function handles it through a recursive call. """
        if isinstance(placement_component, PyArchitecture):
            return PrintableFrozenSet(placement_component.devices)
        elif isinstance(placement_component, PyDevice):
            return [placement_component]
        elif isinstance(placement_component, Collection):
            unpacked_devices = []
            for c in placement_component:
                # If a device specified by users does not exit 
                # and was not registered to the Parla runtime,
                # its instance is set to None and should be
                # ignored.
                if c is None:
                    continue
                tmp_unpacked_devices = self.unpack_devices(c)
                if isinstance(c, Tuple):
                    # Multi-device placement is specified
                    # through a nested list. Therefore, each nested list in the
                    # placement specifies a single placement for
                    # a task, and multiple lists allow the task
                    # mapper to choose one of those lists as resource
                    # requirements depending on device status.
                    unpacked_devices += [tmp_unpacked_devices]
                elif isinstance(c, PyArchitecture):
                    # Architecture placement means one placement
                    # of the entire devices in the architecture.
                    # For example, if `gpu` is specified, all gpu devices
                    # become target candidate devices and one of them
                    # might be chosen.
                    # To distinguish architecture placement from others,
                    # it is converted to a frozen set of the entire devices.
                    unpacked_devices.append(tmp_unpacked_devices)
                else:
                    unpacked_devices += tmp_unpacked_devices
            return unpacked_devices

    def get_devices_from_placement(self, placement):
        """ Unpack placement and return device objects that are specified
            (or implied) through the placement argument of @spawn.
            If None is passed to the placement, all devices exiting
            in the current system become candidates of the placement. """
        if placement is not None:
            ps = placement if isinstance(placement, Iterable) else [placement]
            return self.unpack_devices(ps)
        else:
            return self.get_all_devices()

    def construct_single_device_reqs(self, d, r):
        return DeviceResourceRequirement(device=d, res_req=r)

    def construct_multi_device_reqs(self, devices):
        reqs = []
        for d in devices:
            if isinstance(d, PrintableFrozenSet):
                set_reqs = []
                for set_d in d:
                    set_reqs.append(self.construct_single_device_reqs(set_d, DeviceResource()))
                reqs.append(PrintableFrozenSet(set_reqs))
            else:
                reqs.append(self.construct_single_device_reqs(d, DeviceResource()))
        return reqs

    def construct_single_arch_reqs(self, archs):
        reqs = []
        for d in archs:
            reqs.append(self.construct_single_device_reqs(d, DeviceResource()))
        return frozenset(reqs)

    def get_device_reqs_from_placement(self, placement):
        device_candidates = self.get_devices_from_placement(placement)
        assert(isinstance(device_candidates, List))
        device_reqs = []
        for dc in device_candidates:
            if isinstance(dc, List):
                # TODO(hc): just architecture specification didn't make pairs.
                device_reqs.append(self.construct_multi_device_reqs(dc))
            elif isinstance(dc, FrozenSet):
                device_reqs.append(self.construct_single_arch_reqs(dc))
            else:
                device_reqs.append(self.construct_single_device_reqs(dc, DeviceResource()))
        return device_reqs
