import csv
import time
import argparse

from parla import Parla, spawn, TaskSpace, sleep_nogil, sleep_gil
from parla.cython.device_manager import cpu
from parla.cython.tasks import wait, AtomicTaskList, AtomicTaskSpace
import nvtx

free_sleep = sleep_nogil
lock_sleep = sleep_gil

TaskSpace = AtomicTaskSpace

parser = argparse.ArgumentParser()
parser.add_argument("-workers", type=int, default=1)
parser.add_argument("-n", type=int, default=3)
parser.add_argument("-t", type=int, default=10)
parser.add_argument("-accesses", type=int, default=10)
parser.add_argument("-frac", type=float, default=0)
parser.add_argument('-sweep', type=int, default=0)
parser.add_argument('-verbose', type=int, default=0)
parser.add_argument('-empty', type=int, default=0)
args = parser.parse_args()


def spawn_task(T, i, t, cost):

    @spawn(T[i], placement=[cpu[{'vcus': cost}]])
    def task1():
        nvtx.push_range(message="TASK", color="blue", domain="application")
        if t < 1000:
            return None

        if args.verbose:
            inner_start_t = time.perf_counter()
            print("Task", i, " | Start", flush=True)

        free_sleep(t)

        if args.verbose:
            inner_end_t = time.perf_counter()
            print("Task", i, " | Inner Time: ",
                  inner_end_t - inner_start_t, flush=True)
        nvtx.pop_range(domain="application")

        spawn_task(T, 2*i, t//2, cost)
        spawn_task(T, 2*i+1, t//2, cost)


def main(workers, n, t, accesses, frac):

    @spawn(vcus=0)
    async def task1():
        cost = 1/workers

        start_t = time.perf_counter()
        T = TaskSpace("T")

        nvtx.push_range(message="LAUNCH TASKS", color="green",
                        domain="application")

        spawn_task(T, 1, t, cost) 

        nvtx.pop_range(domain="application")

        barrier_start_t = time.perf_counter()
        #collection = T[:n]
        mid_t = time.perf_counter()
        wait(T)
        #await T
        barrier_end_t = time.perf_counter()

        end_t = time.perf_counter()
        elapsed_t = end_t - start_t

        print("Time to make collection:", mid_t - barrier_start_t)
        
        print(', '.join([str(workers), str(n), str(t), str(
            accesses), str(frac), str(elapsed_t)]), flush=True)
        #print(n/elapsed_t, flush=True)

    # @spawn()
    # def test():
    #    print("HELLO", flush=True)


if __name__ == "__main__":

    print(', '.join([str('workers'), str('n'), str('task_time'), str(
        'accesses'), str('frac'), str('total_time')]), flush=True)
    if not args.sweep:
        with Parla():
            main(args.workers, args.n, args.t, args.accesses, args.frac)
    else:
        for task_time in [1000]:
            for accesses in [1]:
                for nworkers in range(1, args.workers):
                    for frac in [0]:
                        with Parla():
                            main(nworkers, args.n, task_time, accesses, frac)
