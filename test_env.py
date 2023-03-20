from parla import create_env, DeviceManager
from parla.cython.device_manager import cuda
import cupy as cp
from parla.common.globals import _Locals as Locals
from parla.common.globals import get_current_devices, get_active_device, get_current_stream, get_current_context, get_scheduler, get_device_manager

device_manager = DeviceManager(dev_config="devices_sample.YAML")


class placeholder:
    def __init__(self, device_manager):
        self.device_manager = device_manager


Locals.push_scheduler(placeholder(device_manager))

print(device_manager.stream_pool)
context = create_env([cuda(0), cuda(1)])

with context as outer:

    @outer.parallel()
    def f(inner):
        print("outer: ", outer, flush=True)
        print("inner: ", inner, flush=True)
        print("streams: ", inner.streams, flush=True)
        print("cupy: ", cp.cuda.get_current_stream(), cp.cuda.Device())

context.finalize()
print(device_manager.stream_pool)
