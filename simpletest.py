import argparse

from parla import Parla, spawn, TaskSpace, parray, gpu_sleep_nogil
from parla.cython.device_manager import gpu
from parla.common.globals import get_current_devices
import numpy as np
# from sleep.core import bsleep

bsleep = gpu_sleep_nogil
CYCLES = 10000000000


parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str)
args = parser.parse_args()


def main(T):
    a = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6],
                 [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
    a = parray.asarray(a, name='A')

    @spawn(T[0], placement=[(gpu(0), gpu(1))])
    def task1():
        devices = get_current_devices()

        print(T[0], "running on ", devices)

        for device in devices:
            with device:
                print("+ 1 HELLO INNER", flush=True)
                bsleep(device.gpu_id, CYCLES, device.stream.stream)
                print("- 1 HELLO INNER", flush=True)

    @spawn(T[1], dependencies=[T[0]], placement=[gpu(2)])
    def task2():
        devices = get_current_devices()

        for device in devices:
            with device:
                print("+ 0 HELLO INNER", flush=True)
                bsleep(device.gpu_id, CYCLES, device.stream.stream)
                print("- 0 HELLO INNER", flush=True)


if __name__ == "__main__":
    with Parla(dev_config_file=args.dev_config):
        T = TaskSpace("T")
        main(T)
