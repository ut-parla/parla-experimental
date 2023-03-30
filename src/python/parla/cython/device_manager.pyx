from parla.cython import device
from parla.cython.device cimport Device
from parla.common.globals import DeviceType, cupy

from typing import FrozenSet, Collection, Iterable, Set, Tuple, List

import os
import psutil
import yaml

VCU_BASELINE=1000

PyDevice = device.PyDevice
PyCUDADevice = device.PyCUDADevice
PyCPUDevice = device.PyCPUDevice
PyInvalidDevice = device.PyInvalidDevice
PyArchitecture = device.PyArchitecture
PyCUDAArchitecture = device.PyCUDAArchitecture
PyCPUArchitecture = device.PyCPUArchitecture
DeviceResource = device.DeviceResource
DeviceResourceRequirement = device.DeviceResourceRequirement
Stream = device.Stream
CupyStream = device.CupyStream
CUPY_ENABLED = device.CUPY_ENABLED

# Architecture declaration.
# To use these in the placement of @spawn,
# declare these as global variables.
cuda = PyCUDAArchitecture()
cpu = PyCPUArchitecture()



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
        return self.cpp_device_manager_.globalid_to_parrayid(global_dev_id)

    cpdef parrayid_to_globalid(self, parray_dev_id):
        return self.cpp_device_manager_.parrayid_to_globalid(parray_dev_id)

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

class StreamPool:

    def __init__(self, device_list, per_device=4):

        if CUPY_ENABLED:
            StreamClass = CupyStream 
        else:
            StreamClass = Stream

        self._device_list = device_list
        self._per_device = per_device
        self._pool = {}

        for device in self._device_list:
            self._pool[device] = []
            
            with device.device as d:
                for i in range(self._per_device):
                    self._pool[device].append(StreamClass(device=d))

    def get_stream(self, device):
        if len(self._pool[device]) == 0:
            #Create a new stream if the pool is empty.
            return Stream(device=device)
        return self._pool[device].pop()

    def return_stream(self, stream):
        self._pool[stream.device].append(stream)

    def __summarize__(self):
        summary  = ""
        for device in self._device_list:
            summary += f"({device} : {len(self._pool[device])})"

        return summary

    def __repr__(self):
        return f"StreamPool({self.__summarize__()})"


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
        # TODO(wlr): I made this a dict so that device manager doesn't have to have all arch types.
        self.py_registered_archs = {}
        self.registered_devices = []

        # TODO(hc): pack those config. to a data class.
        # Initialize Devices
        if dev_config == None or dev_config == "":
            self.register_cpu_devices()
            self.register_cuda_devices_cupy()
        else:
            self.parse_config_and_register_devices(dev_config)
        self.register_devices_to_cpp()

        # Initialize Device Hardware Queues
        self.stream_pool = StreamPool(cuda.devices)

    def __dealloc__(self):
        for arch in self.py_registered_archs:
            for dev in arch.devices:
                del dev

    # TODO(hc): this postfix *_cupy() looks confusing.
    def register_cuda_devices_cupy(self):
        if cupy is not None:
            try:
                num_of_gpus = cupy.cuda.runtime.getDeviceCount()
            except cupy.cuda.runtime.CUDARuntimeError:
                num_of_gpus = 0
        else:
            num_of_gpus = 0

        if num_of_gpus > 0:
            self.py_registered_archs[cuda] = cuda
            for dev_id in range(num_of_gpus):
                gpu_dev = cupy.cuda.Device(dev_id)
                mem_info = gpu_dev.mem_info # tuple of free and total memory
                                            # in bytes.
                mem_sz = long(mem_info[1])
                py_cuda_device = PyCUDADevice(dev_id, mem_sz, VCU_BASELINE)
                cuda.add_device(py_cuda_device)
                self.registered_devices.append(py_cuda_device)
        else:
            # It is possible that the current system does not have CUDA devices.
            # But users can still specify `cuda` to task placement.
            # To handle this case, we add a CPU device as the CUDA architecture
            # type (So, Parla assumes that the target system must be equipped
            # with at least one CPU core).
            self.register_cpu_devices(cuda)

    def register_cpu_devices(self, register_to_cuda: bool = False):
        if register_to_cuda:
            cuda.add_device(cpu(0))
            self.registered_devices.append(cpu(0))
        else:
            # Get the number of usable CPUs from this process.
            # This might not be equal to the number of CPUs in the system.
            num_cores = len(os.sched_getaffinity(0))
            mem_sz = long(psutil.virtual_memory().total)
            py_cpu_device = PyCPUDevice(0, mem_sz, VCU_BASELINE)
            self.py_registered_archs[cpu] = cpu
            cpu.add_device(py_cpu_device)
            self.registered_devices.append(py_cpu_device)

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
                self.py_registered_archs[cpu] = cpu
                cpu_mem_sz = parsed_configs["CPU"]["mem_sz"]
                py_cpu_device = PyCPUDevice(0, cpu_mem_sz, VCU_BASELINE) 
                cpu.add_device(py_cpu_device)
                self.registered_devices(py_cpu_device)
            gpu_num_devices = parsed_configs["GPU"]["num_devices"]
            if gpu_num_devices > 0:
                self.py_registered_archs[cuda] = cuda
                gpu_mem_sizes = parsed_configs["GPU"]["mem_sz"]
                assert(gpu_num_devices == len(gpu_mem_sizes)) 
                for dev_id in range(gpu_num_devices):
                    py_cuda_device = PyCUDADevice(dev_id, \
                                                  gpu_mem_sizes[dev_id], \
                                                  VCU_BASELINE)
                    cuda.add_device(py_cuda_device)
                    self.registered_devices.append(py_cuda_device)

    def get_all_devices(self):
        return self.registered_devices

    def get_all_architectures(self):
        return self.py_registered_archs.values()

    def is_multidevice_placement(self, placement_tuple):
        if len(placement_tuple) == 2 and \
                isinstance(placement_tuple[1], DeviceResource):
            return False
        return True

    def construct_single_device_requirements(self, dev, res_req = None):
        res_req_ = res_req if res_req is not None else DeviceResource()
        if isinstance(dev, PyInvalidDevice):
            # If the specified device does not exist
            # in the current system, replace it with CPU.
            dev = cpu(0)
        return DeviceResourceRequirement(dev, res_req_)

    def construct_single_architecture_requirements(self, arch, res_req = None):
        arch_reqs = []
        res_req_ = res_req if res_req is not None else DeviceResource()
        for d in arch.devices:
            arch_reqs.append(self.construct_single_device_requirements(
                  d, res_req_))
        return PrintableFrozenSet(arch_reqs)

    def construct_resource_requirements(self, placement_component):
        if isinstance(placement_component, Tuple) and \
              not self.is_multidevice_placement(placement_component):
                # In this case, the placement component consists of
                # Device or Architecture, with its resource requirement.
                placement, req = placement_component
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
            return self.construct_single_architecture_requirements(
                placement_component)
        elif isinstance(placement_component, PyDevice):
            return self.construct_single_device_requirements(
                placement_component)
        else:
            raise TypeError("Incorrect placement")


    def unpack_placements(self, placement_components):
        """ Unpack a placement parameter and return a list of
            a pair of devices and requirements in a proper hierarchy structure.
            Placements (from @spawn) could be collections, for
            multi-device placements, a pair of architecture and
            resource requirement, or a pair of device and resource requirement.
        """
        assert(isinstance(placement_components, List) or \
            isinstance(placement_components, Tuple))
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
                unpacked_devices.append(self.unpack_placements(c))
            else:
                unpacked_devices.append(self.construct_resource_requirements(c))
        return unpacked_devices

    def get_device_reqs_from_placement(self, placement):
        """ Unpack placement and return device objects that are specified
            (or implied) through the placement argument of @spawn.
            If None is passed to the placement, all devices exiting
            in the current system become candidates of the placement. """
        if placement is not None:
            ps = placement if isinstance(placement, Iterable) else [placement]
            return self.unpack_placements(ps)
        else:
            return [DeviceResourceRequirement(d, DeviceResource()) \
                    for d in self.get_all_devices()]

    def globalid_to_parrayid(self, global_dev_id):
        return self.cy_device_manager.globalid_to_parrayid(global_dev_id)

    def parrayid_to_globalid(self, parray_dev_id):
        return self.cy_device_manager.parrayid_to_globalid(parray_dev_id)
