import argparse

from parla import Parla, spawn, TaskSpace, sleep_nogil, parray
from parla.cython.device_manager import cpu, cuda
from parla.common.globals import get_current_devices
import numpy as np
# from sleep.core import bsleep

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

    """
    @spawn(T[0])
    def task1():
        print("+HELLO OUTER 0", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 0", flush=True)

    """
# @spawn(T[0], placement=[(cpu[{"vcus":20, "memory":40000}], cuda[{"vcus":1, "memory":20}])])
    @spawn(T[0], placement=[(cpu[{"vcus": 20, "memory": 40000}], cuda[{"vcus": 1, "memory": 20}]), (cpu(0), cuda(1)[{"vcus": 10, "memory": 2000}])], \
          input=[a,b], output=[c], inout=[d])
# @spawn(T[0], placement=[(cuda[{"vcus":20, "memory":40000}])])
    def task1():
        print("+HELLO OUTER 0", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 0", get_current_devices(), flush=True)

    @spawn(T[1], placement=[cpu[{"vcus":0, "memory":1000}]], dependencies=[T[0]], inout=[c, d])
    def task2():
        print("+HELLO OUTER 1", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 1", get_current_devices(), flush=True)

    @spawn(T[2], placement=[(cuda(1), cuda(2))], vcus=0, input=[a, b])
    def task3():
        print("+HELLO OUTER 2", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 2", get_current_devices(), flush=True)
    # @spawn()
    # def test():
    #    print("HELLO", flush=True)


if __name__ == "__main__":
    with Parla(dev_config_file=args.dev_config):
        T = TaskSpace("T")
        main(T)
