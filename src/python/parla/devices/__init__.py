from ..cython import device_manager
from ..cython import device

DeviceManager = device_manager.PyDeviceManager
Stream = device.Stream

cpu = device_manager.cpu
gpu = device_manager.gpu

__all__ = ["DeviceManager", "Stream", "cpu", "gpu"]
