import nvtx
from sleep.core import bsleep as free_sleep
import time
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-workers", type=int, default=1)
parser.add_argument("-n", type=int, default=1000)
parser.add_argument("-t", type=int, default=1000)
parser.add_argument("-accesses", type=int, default=10)
parser.add_argument("-frac", type=float, default=0)
parser.add_argument('-sweep', type=int, default=0)
parser.add_argument('-verbose', type=int, default=0)
parser.add_argument('-empty', type=int, default=0)
args = parser.parse_args()

if use_old := os.getenv("USE_OLD_PARLA"):
    from parla import Parla
    from parla.cuda import gpu
    from parla.tasks import *
else:
    from parla import Parla, spawn, TaskSpace
    from parla.cython.device_manager import cpu


def main(workers, n, t):

    @spawn(vcus=0)
    async def task1():

        cost = 1000.0/workers
        start_t = time.perf_counter()
        T = TaskSpace("T")

        nvtx.push_range(message="launch", color="green",
                        domain="application")

        for i in range(n):
            @spawn(T[i], placement=cpu, vcus=cost)
            def task1():
                nvtx.push_range(message="task", color="blue",
                                domain="application")
                free_sleep(t)
                nvtx.pop_range(domain="application")

        nvtx.pop_range(domain="application")

        await T

        end_t = time.perf_counter()
        elapsed_t = end_t - start_t

        print(', '.join([str(workers), str(n),
              str(t), str(elapsed_t)]), flush=True)


if __name__ == "__main__":
    main(args.workers, args.n, args.t)
