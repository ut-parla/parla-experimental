from parla import Parla
from parla.tasks import spawn, get_current_context
from parla.array import asarray
from parla.devices import gpu

import numpy as np
import cupy as cp
import crosspy as xp
import time
from cufftmg import handler

size = 21000

def main():

    N = size
    ngpus = 1
    x = np.arange(N**2, dtype=np.float64) * 2 + 1
    y = (np.arange(N**2, dtype=np.float64) + 1) * 2
    x = np.reshape(x, (N, N))
    y = np.reshape(y, (N, N))
    x = x + 1j * y

    device_ids = [i for i in range(ngpus)]
    stream_list = []
    cupy_list = []

    step = N // len(device_ids)

    for i in device_ids:
        with cp.cuda.Device(i):
            local_copy = x[i*step:(i+1)*step, :]
            local_copy = cp.asarray(local_copy)
            stream_list.append(cp.cuda.get_current_stream())
            cupy_list.append(local_copy)

    class PArrayManager(xp.utils.wrapper.DynamicObjectManager):
        def wrap(self, array):
            return asarray(array)
        
        def get_device(self, id):
            return get_current_context().devices[id]

    array = xp.array(cupy_list, axis=0, data_manager=PArrayManager())
    #array = xp.array(cupy_list, axis=0, wrapper=asarray)
    place = tuple([gpu(i) for i in range(ngpus)])
    @spawn(placement=[place], input=[array], vcus=1)
    def task():
        context = get_current_context()
        print(context)
        gpu_ids = context.gpu_ids 
        streams = []

        for id in gpu_ids:
            with cp.cuda.Device(id):
                stream = cp.cuda.Stream()
                streams.append(stream)

        for stream in streams:
            stream.synchronize()
        #print(streams)
        start_t = time.perf_counter()
        h = handler()
        h.configure(device_ids=gpu_ids, streams=streams, size=size)
        h.fft2(array)
        h.ifft2(array)

        for stream in streams:
            stream.synchronize()

        end_t = time.perf_counter()

        print("Elapsed: ", end_t - start_t, flush=True)


if __name__ == "__main__":
    with Parla():
        main()
