from parla.common.global_enums import DeviceType 

import nvidia_smi

from dataclasses import dataclass

@dataclass
class DeviceConfig:
    """ Class for keeping track of device resources. """

    # Device name.
    dev_name: str
    # Device type (e.g., CPU or CUDA).
    dev_type: DeviceType
    # Id 0 is assigned to the first device of
    # each device type.
    dev_id: int
    num_vcus: int
    mem_sz: int


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
