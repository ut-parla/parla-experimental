import argparse

from parla import Parla, spawn, TaskSpace, sleep_nogil
from parla.cython.device_manager import cpu, cuda
# from sleep.core import bsleep

bsleep = sleep_nogil

parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str)
args = parser.parse_args()

def main(T):


    @spawn(T[0], placement=[{cpu, cuda}, {cpu(0), cuda(1)}], vcus=0)
    def task1():
        print("+HELLO OUTER 0", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 0", flush=True)

    @spawn(T[1], placement=[cpu], vcus=0, dependencies=[T[0]])
    def task2():
        print("+HELLO OUTER 1", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 1", flush=True)

    @spawn(T[2], placement=[[cuda(1), cuda(2)]], vcus=0)
    def task3():
        print("+HELLO OUTER 2", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 2", flush=True)

    # @spawn()
    # def test():
    #    print("HELLO", flush=True)


if __name__ == "__main__":
    with Parla(dev_config_file=args.dev_config):
        T = TaskSpace("T")
        main(T)
