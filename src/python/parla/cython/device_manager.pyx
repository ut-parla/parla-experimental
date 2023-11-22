# cython: language_level=3
# cython: language=c++
"""!
@file device_manager.pyx
@brief Contains the cython wrapper and python layer DeviceManager and StreamPool classes.
"""

from . import device
from .device cimport Device
from ..common.globals import DeviceType, cupy, VCU_BASELINE

from typing import Iterable, Tuple, List

import os
import psutil
import yaml

PyDevice = device.PyDevice
PyGPUDevice = device.PyGPUDevice
PyCPUDevice = device.PyCPUDevice
PyArchitecture = device.PyArchitecture
ImportableGPUArchitecture = device.ImportableGPUArchitecture
ImportableCPUArchitecture = device.ImportableCPUArchitecture
PyGPUArchitecture = device.PyGPUArchitecture
PyCPUArchitecture = device.PyCPUArchitecture
DeviceResource = device.DeviceResource
DeviceResourceRequirement = device.DeviceResourceRequirement
Stream = device.Stream
CupyStream = device.CupyStream
CUPY_ENABLED = device.CUPY_ENABLED

# Importable architecture declarations
gpu = ImportableGPUArchitecture()
cpu = ImportableCPUArchitecture()

from . import stream_pool

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
        self.cpp_device_manager_.register_device(cpp_device)

    cpdef print_registered_devices(self):
        self.cpp_device_manager_.print_registered_devices()

    cpdef globalid_to_parrayid(self, global_dev_id):
        return g2p(global_dev_id)

    cpdef parrayid_to_globalid(self, parray_dev_id):
        return p2g(parray_dev_id)

    cdef DeviceManager* get_cpp_device_manager(self):
        return self.cpp_device_manager_

    cpdef free_memory(self, parray_dev_id, memory_size):
        self.cpp_device_manager_.free_memory_by_parray_id(parray_dev_id, memory_size)


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

# TODO(wlr):  - Allow device manager to initialize non-contiguous gpu ids. 
# TODO(wlr):  - Provide a way to iterate over these real device ids
           

class PyDeviceManager:
    """
    A device manager manages device objects and provides their information.
    The Parla runtime should access device information through a device manager.
    A single device manager for each Python and C++ is created and held
    by each schedulers.
    """

    def __init__(self, dev_config = None):
        self.cy_device_manager = CyDeviceManager()

        self.py_registered_archs = {}
        self.registered_devices = []

        if CUPY_ENABLED:
            try:
                self.num_real_gpus = cupy.cuda.runtime.getDeviceCount()
            except cupy.cuda.runtime.CUDARuntimeError:
                self.num_real_gpus = 0
        else:
            self.num_real_gpus = 0

        # Initialize Devices
        if dev_config is None or dev_config == "":
            self.register_cpu_devices()
            self.register_cupy_gpu_devices()
        else:
            self.parse_config_and_register_devices(dev_config)
        # self.register_devices_to_cpp()

        # Initialize Device Hardware Queues
        self.stream_pool = stream_pool.CyStreamPool(self.get_devices(DeviceType.GPU))

    def __dealloc__(self):
        for arch in self.py_registered_archs:
            for dev in arch.devices:
                del dev

    def register_cupy_gpu_devices(self):
        """
        This function adds cupy GPU devices.
        """
        # TODO(hc): Later, it will extend a GPU architecture type, like ones
        #           from AMD.
        if cupy is not None:
            try:
                num_of_gpus = cupy.cuda.runtime.getDeviceCount()
            except cupy.cuda.runtime.CUDARuntimeError:
                num_of_gpus = 0
        else:
            num_of_gpus = 0

        if num_of_gpus > 0:
            gpu_arch = PyGPUArchitecture()
            self.py_registered_archs[gpu] = gpu_arch

            for dev_id in range(num_of_gpus):
                gpu_dev = cupy.cuda.Device(dev_id)
                mem_info = gpu_dev.mem_info  # tuple of free and total memory (in bytes)
                mem_sz = long(mem_info[1])
                py_cuda_device = PyGPUDevice(dev_id, mem_sz, VCU_BASELINE)

                # Add device to the architecture
                gpu_arch.add_device(py_cuda_device)

                # Add device to the device manager (list of devices)
                self.registered_devices.append(py_cuda_device)

                # Register device to the C++ runtime
                cy_device = py_cuda_device.get_cy_device()
                self.cy_device_manager.register_device(cy_device)

    def register_cpu_devices(self, register_to_cuda: bool = False):
        num_cores = os.getenv("PARLA_NUM_CORES")
        if num_cores:
            num_cores = int(num_cores)
        else:
            num_cores = psutil.cpu_count(logical=False)
        if num_cores == 0:
            raise RuntimeError("No CPU cores available for Parla.")

        mem_sz = os.getenv("PARLA_CPU_MEM")
        if mem_sz:
            mem_sz = int(mem_sz)
        else:
            mem_sz = long(psutil.virtual_memory().total)
        
        py_cpu_device = PyCPUDevice(0, mem_sz, VCU_BASELINE)

        cpu_arch = PyCPUArchitecture()
        self.py_registered_archs[cpu] = cpu_arch
        cpu_arch.add_device(py_cpu_device)

        self.registered_devices.append(py_cpu_device)
        cy_device = py_cpu_device.get_cy_device()
        self.cy_device_manager.register_device(cy_device)
        
    def register_devices_to_cpp(self):
        """
        Register devices to the both Python/C++ runtime.
        """
        current_idx = 0
        for py_arch in self.py_registered_archs:
            for py_device in py_arch.devices:
                cy_device = py_device.get_cy_device()
                self.cy_device_manager.register_device(cy_device)
                py_device.global_id = current_idx 
                current_idx += 1

    def print_registered_devices(self):
        print("Python devices:", flush=True)
        for dev in self.py_registered_archs:
            print(f"\t Registered device: {dev}", flush=True)
        self.cy_device_manager.print_registered_devices()

    def get_cy_device_manager(self):
        return self.cy_device_manager

    def get_num_gpus(self) -> int:
        return len(self.py_registered_archs[gpu].devices)

    def get_devices(self, architecture_type):
        if architecture_type not in self.py_registered_archs:
            return []
        return self.py_registered_archs[architecture_type].devices

    def get_num_cpus(self) -> int:
        return len(self.py_registered_archs[cpu].devices)

    def get_architecture(self, arch_type) -> PyArchitecture:
        return self.py_registered_archs[arch_type]

    def parse_config_and_register_devices(self, yaml_config):
        with open(yaml_config, "r") as f:
            parsed_configs = yaml.safe_load(f)
            # Parse CPU device information.
            cpu_num_cores = parsed_configs["CPU"]["num_cores"]
            if cpu_num_cores > 0:
                cpu_arch = PyCPUArchitecture()
                self.py_registered_archs[cpu] = cpu_arch
                cpu_mem_sz = parsed_configs["CPU"]["mem_sz"]
                py_cpu_device = PyCPUDevice(0, cpu_mem_sz, VCU_BASELINE) 
                cpu_arch.add_device(py_cpu_device)
                self.registered_devices.append(py_cpu_device)
                cy_device = py_cpu_device.get_cy_device()
                self.cy_device_manager.register_device(cy_device)

            num_of_gpus = parsed_configs["GPU"]["num_devices"]
            if num_of_gpus > 0:
                gpu_arch = PyGPUArchitecture()
                self.py_registered_archs[gpu] = gpu_arch
                gpu_mem_sizes = parsed_configs["GPU"]["mem_sz"]
                assert(num_of_gpus == len(gpu_mem_sizes)) 
                
                for dev_id in range(num_of_gpus):

                    if self.num_real_gpus > 0:
                        py_cuda_device = PyGPUDevice(
                                                dev_id % self.num_real_gpus,
                                                gpu_mem_sizes[dev_id],
                                                VCU_BASELINE
                                            )
                    
                    else:
                        py_cuda_device = PyCPUDevice(
                                                dev_id,
                                                gpu_mem_sizes[dev_id],
                                                VCU_BASELINE
                                            )

                    gpu_arch.add_device(py_cuda_device)
                    self.registered_devices.append(py_cuda_device)
                    cy_device = py_cuda_device.get_cy_device()
                    self.cy_device_manager.register_device(cy_device)

    def get_all_devices(self):
        return self.registered_devices

    def get_all_architectures(self):
        return self.py_registered_archs.values()

    def is_multidevice_placement(self, placement_tuple):
        if len(placement_tuple) == 2 and \
                isinstance(placement_tuple[1], DeviceResource):
            return False
        return True

    def construct_single_device_requirements(self, dev, res_req):
        return DeviceResourceRequirement(dev, res_req)

    def construct_single_architecture_requirements(self, arch, res_req):
        arch_reqs = []
        for d in arch.devices:
            arch_reqs.append(self.construct_single_device_requirements(
                  d, res_req))
        return PrintableFrozenSet(arch_reqs)

    def construct_resource_requirements(self, placement_component, vcus, memory):
        if isinstance(placement_component, Tuple) and not self.is_multidevice_placement(placement_component):
            # In this case, the placement component consists of
            # Device or Architecture, with its resource requirement.
            placement, req = placement_component
            req.memory = req.memory if req.memory is not None else  \
                (0 if memory is None else memory)
            req.vcus = req.vcus if req.vcus is not None else  \
                (0 if vcus is not None else vcus)
            # If a device specified by users does not exit 
            # and was not registered to the Parla runtime,
            # use CPU instead.
            if isinstance(placement, PyArchitecture):
                # Architecture placement means that the task mapper
                # could choose one of the devices in the specified
                # architecture.
                # For example, if `gpu` is specified, all gpu devices
                # become target candidate devices and one of them
                # might be chosen as the final placement for a task.
                # To distinguish architecture placement from others,
                # it is converted to a frozen set of the entire devices.
                return self.construct_single_architecture_requirements(
                    placement, req)
            elif isinstance(placement, PyDevice):
                return self.construct_single_device_requirements(
                    placement, req)
        elif isinstance(placement_component, PyArchitecture):
            vcus = vcus if vcus is not None else 0
            memory = memory if memory is not None else 0
            res_req = DeviceResource(memory, vcus)
            return self.construct_single_architecture_requirements(
                placement_component, res_req)
        elif isinstance(placement_component, PyDevice):
            vcus = vcus if vcus is not None else 0
            memory = memory if memory is not None else 0
            res_req = DeviceResource(memory, vcus)
            return self.construct_single_device_requirements(
                placement_component, res_req)
        else:
            raise TypeError("Incorrect placement")

    def unpack_placements(self, placement_components, vcus, memory):
        """ Unpack a placement parameter and return a list of
            a pair of devices and requirements in a proper hierarchy structure.
            Placements (from @spawn) could be collections, for
            multi-device placements, a pair of architecture and
            resource requirement, or a pair of device and resource requirement.
        """
        assert(isinstance(placement_components, List) or isinstance(placement_components, Tuple))
        # Multi-device resource requirement or
        # a list of devices, architectures, or multi-device 
        # requirements.
        unpacked_devices = []
        for c in placement_components:
            if isinstance(c, Tuple) and self.is_multidevice_placement(c):
                # Multi-device placement is specified
                # through a nested tuple of the placement API.
                # Which means that, each nested tuple in the
                # placement specifies a single placement for
                # a task. The placement API allows multiple tuples for
                # multi-device placements (e.g., placement=[(), (), ..]),
                # and the task mapper chooses one of those options
                # as the target requirement based on device states.
                # In this case, recursively call this function and
                # construct a list of member devices and their resource
                # requirements to distinguish them from other flat
                # resource requirements.
                unpacked_devices.append(self.unpack_placements(c, vcus, memory))
            else:
                unpacked_devices.append(self.construct_resource_requirements(c, vcus, memory))
        return unpacked_devices

    def get_device_reqs_from_placement(self, placement, vcus, memory):
        """ Unpack placement and return device objects that are specified
            (or implied) through the placement argument of @spawn.
            If None is passed to the placement, all devices exiting
            in the current system become candidates of the placement. """
        # Placement cannot be None since it is set to a list of the whole 
        # devices.
        assert placement is not None
        ps = placement if isinstance(placement, Iterable) else [placement]
        return self.unpack_placements(ps, vcus, memory)

    def globalid_to_parrayid(self, global_dev_id):
        return self.cy_device_manager.globalid_to_parrayid(global_dev_id)

    def parrayid_to_globalid(self, parray_dev_id):
        return self.cy_device_manager.parrayid_to_globalid(parray_dev_id)

    def free_memory(self, parray_dev_id, size):
        """
        Free memory of the device with the given device id.
        Note: The device id is the PArray id of the device, not the internal global Parla id.
        Frees the memory on both the devices Mapped and Reserved Pools.
        """
        self.cy_device_manager.free_memory(parray_dev_id, size)
