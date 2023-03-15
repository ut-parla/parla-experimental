from parla import TaskEnvironment, GPUEnvironment, StreamPool, Locals, create_env, DeviceManager, GlobalStreamPool
from parla.cython.device_manager import cpu, cuda
import cupy as cp


device_manager = DeviceManager()

pool = StreamPool([cuda(0), cuda(1)])

GlobalStreamPool.set(pool)


print(GlobalStreamPool.get())

context = create_env([cuda(0), cuda(1)])

with context as outer:

    with context[0] as inner:
        print("outer: ", outer, flush=True)
        print("inner: ", inner, flush=True)
        print("streams: ", inner.streams, flush=True)
        print("cupy: ", cp.cuda.get_current_stream(), cp.cuda.Device())

    with context[1] as inner:
        print("outer: ", outer, flush=True)
        print("inner: ", inner, flush=True)
        print("streams: ", inner.streams, flush=True)
        print("cupy: ", cp.cuda.get_current_stream(), cp.cuda.Device())

print(pool)


