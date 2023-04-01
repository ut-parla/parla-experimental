import argparse

from parla import Parla, spawn, TaskSpace, sleep_nogil, parray
from parla.cython.device_manager import cuda
from parla.common.globals import get_current_devices
import numpy as np
# from sleep.core import bsleep
import cupy as cp
bsleep = sleep_nogil

parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str)
args = parser.parse_args()


def main(T):
    a = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
    b = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
    c = np.array([[2, 3, 5, 6, 7], [2, 3, 5, 6, 7], [2, 3, 5, 6, 7], [2, 3, 5, 6, 7]])
    d = np.array([[2, 3, 5, 6, 7], [2, 3, 5, 6, 7], [2, 3, 5, 6, 7], [2, 3, 5, 6, 7]])
    a = parray.asarray(a)
    b = parray.asarray(b)
    c = parray.asarray(c)
    d = parray.asarray(d)


    for i in range(10):
        @spawn(T[1+10*i], placement=[(cuda(0))])
        def task1():
            print("+HELLO OUTER 0", flush=True)
            bsleep(1000)
            print("-HELLO OUTER 0", get_current_devices(), cp.cuda.runtime.getDevice(), flush=True)

        @spawn(T[2+10*i], placement=[(cuda(1))])
        def task2():
            print("+HELLO OUTER 1", flush=True)
            bsleep(1000)
            print("-HELLO OUTER 1", get_current_devices(), cp.cuda.runtime.getDevice(), flush=True)


        @spawn(T[3+10*i], placement=[(cuda(2))])
        def task3():
            print("+HELLO OUTER 2", flush=True)
            bsleep(1000)
            print("-HELLO OUTER 2", get_current_devices(), cp.cuda.runtime.getDevice(), flush=True)


if __name__ == "__main__":
    with Parla():
        T = TaskSpace("T")
        main(T)
