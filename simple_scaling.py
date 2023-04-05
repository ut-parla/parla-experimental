import argparse
import time
from parla import Parla, spawn, TaskSpace, parray, gpu_sleep_nogil
from parla.cython.device_manager import gpu, cpu
from parla.common.globals import get_current_devices
import numpy as np
# from sleep.core import bsleep
import time
bsleep = gpu_sleep_nogil


#CYCLES = 10000000000 #5.525 seconds

CYCLES = 100000000 #.05 seconds


parser = argparse.ArgumentParser()
parser.add_argument("-ngpus", type=int, default=4)
args = parser.parse_args()


def main(T):

    @spawn(placement=cpu)
    async def main_task():

        start_t = time.perf_counter()
        for i in range(100):

            @spawn(T[i], placement=[gpu(i%args.ngpus)[{'vcus': 1000}]])
            def task1():
                devices = get_current_devices()

                for device in devices:
                    with device:
                        #print("+ 1 HELLO INNER", flush=True)
                        internal_start_t = time.perf_counter()
                        bsleep(device.gpu_id, CYCLES, device.stream.stream)
                        device.synchronize()
                        internal_end_t = time.perf_counter()
                        #print("- 1 HELLO INNER. Elapsed: ",
                        #      internal_end_t - internal_start_t, flush=True)

        await T
        end_t = time.perf_counter()
        print("Time elapsed: ", end_t - start_t, flush=True)


if __name__ == "__main__":
    with Parla():
        T = TaskSpace("T")
        main(T)
