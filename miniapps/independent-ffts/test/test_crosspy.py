from parla import Parla
from parla.tasks import spawn, get_current_context
from parla.array import asarray
from parla.devices import gpu

import numpy as np
import cupy as cp
import crosspy as xp

from cufftmg import handler

def main():

    N = 40
    x = np.arange(N**2, dtype=np.float64) * 2 + 1
    y = (np.arange(N**2, dtype=np.float64) + 1) * 2
    x = np.reshape(x, (N, N))
    y = np.reshape(y, (N, N))
    x = x + 1j * y

    device_ids = [0, 1]
    stream_list = []
    cupy_list = []

    step = N // len(device_ids)

    for i in device_ids:
        with cp.cuda.Device(i):
            local_copy = x[i*step:(i+1)*step, :]
            local_copy = cp.asarray(local_copy)
            stream_list.append(cp.cuda.get_current_stream())
            cupy_list.append(local_copy)

    parray1 = asarray(cupy_list[0])
    parray2 = asarray(cupy_list[1])

    class PArrayManager(xp.utils.wrapper.DynamicObjectManager):
        def wrap(self, array):
            return asarray(array)
        
        def get_device(self, id):
            return get_current_context().devices[id]

    array = xp.array(cupy_list, axis=0, data_manager=PArrayManager())
    #array = xp.array(cupy_list, axis=0, wrapper=asarray)

    @spawn(placement=[(gpu(3), gpu(2))], input=[array])
    def task():
        for arr in array.block_view():
            arr.print_overview()


if __name__ == "__main__":
    with Parla():
        main()

