import time
import argparse

import numpy as np
from parla import Parla, spawn, TaskSpace, gpu_sleep_nogil, sleep_nogil
from parla.common.globals import get_current_context
from parla.cython.device_manager import gpu, cpu
from parla.cython.tasks import AtomicTaskSpace
#from parla.utility.execute import _GPUInfo as GPUInfo 

TaskSpace = AtomicTaskSpace

import nvtx

free_sleep = gpu_sleep_nogil


parser = argparse.ArgumentParser()
parser.add_argument("-workers", type=int, default=1)
parser.add_argument("-n", type=int, default=3)
parser.add_argument("-t", type=int, default=10)
parser.add_argument("-accesses", type=int, default=1)
parser.add_argument("-frac", type=float, default=0)
parser.add_argument('-sweep', type=int, default=0)
parser.add_argument('-verbose', type=int, default=0)
parser.add_argument('-empty', type=int, default=0)
parser.add_argument('-devices', type=int, default=1)
parser.add_argument('-ngpus', type=int, default=1)
args = parser.parse_args()

cycles_per_second = 1949802881.4819772
#GPUInfo.update(ticks=int(1889922225*(1/64)))
#cycles_per_second = GPUInfo.get()

count = np.zeros(4, dtype=np.int64)

def main(workers, n, t, accesses, frac):

    @spawn(placement=cpu, vcus=0)
    async def task1():
        ta = t/1e6
        kernel_time = ta / accesses
        free_time = kernel_time * (1 - frac)
        kernel_time * frac

        int(free_time * cycles_per_second)

        start_t = time.perf_counter()
        T = TaskSpace("T")

        nvtx.push_range(message="LAUNCH TASKS", color="green",
                        domain="application")

        for i in range(n):
            @spawn(T[i], placement=[gpu*args.devices], vcus=1)
            def task1():
                nvtx.push_range(message="TASK", color="blue", domain="application")
                if args.empty:
                    return None

                context = get_current_context()
                ids = context.gpu_ids 
                
                if args.verbose:
                    inner_start_t = time.perf_counter()
                    print("Task", i, " | Start: ", ids, flush=True)

                for idx in ids:
                    count[idx]+=1

                for device in context.loop():
                    pass

                    #free_sleep(device_id, free_cycles, stream)

                sleep_nogil(t)

                context.synchronize()


                if args.verbose:
                    inner_end_t = time.perf_counter()
                    print("Task", i, " | Inner Time: ",
                          inner_end_t - inner_start_t, ids, flush=True)
                nvtx.pop_range(domain="application")

        nvtx.pop_range(domain="application")

        time.perf_counter()
        T.wait()
        time.perf_counter()

        end_t = time.perf_counter()
        elapsed_t = end_t - start_t
        
        print(', '.join([str(workers), str(n), str(t), str(
            accesses), str(frac), str(elapsed_t)]), flush=True)

        print(count)
        #print(n/elapsed_t, flush=True)

    # @spawn()
    # def test():
    #    print("HELLO", flush=True)


if __name__ == "__main__":

    print(', '.join([str('workers'), str('n'), str('task_time'), str(
        'accesses'), str('frac'), str('total_time')]), flush=True)
    if not args.sweep:
        for i in range(1):
            with Parla():
                print(gpu, len(gpu))
                main(args.workers, args.n, args.t, args.accesses, args.frac)
    else:
        for task_time in [1000]:
            for accesses in [1]:
                for nworkers in range(1, args.workers):
                    for frac in [0]:
                        with Parla():
                            main(nworkers, args.n, task_time, accesses, frac)
