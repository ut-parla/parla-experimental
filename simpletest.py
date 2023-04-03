import argparse

from parla import Parla, spawn, TaskSpace, sleep_nogil, parray
from parla.cython.device_manager import cpu, gpu
from parla.common.globals import get_current_devices
import numpy as np
# from sleep.core import bsleep

bsleep = sleep_nogil

parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str)
args = parser.parse_args()


def main(T):
    a = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
    a = parray.asarray(a, name='A')

    @spawn(T[0], placement=[gpu(0), gpu(1)])
    def task1():
        print("+HELLO OUTER 0", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 0", flush=True)

    @spawn(T[1], dependencies=[T[0]], placement=[gpu(2)])
    def task2():
        print("+HELLO OUTER 1", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 1", flush=True)


if __name__ == "__main__":
    with Parla(dev_config_file=args.dev_config):
        T = TaskSpace("T")
        main(T)
