import os
import time
import ray
from concurrent.futures import ThreadPoolExecutor

from parla import sleep_nogil, sleep_gil
from parla import sleep_nogil as sleep
import argparse

free_sleep = sleep_nogil
lock_sleep = sleep_gil

os.environ["RAY_DEDUP_LOGS"] = str(0)
os.environ["RAY_LOG_TO_STDERR"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=14)
parser.add_argument('-workers', default=4, type=int)
parser.add_argument('-t', type=int, default=1000)
parser.add_argument('-accesses', type=int, default=1)
parser.add_argument('-verbose', type=int, default=0)
parser.add_argument('-frac', type=float, default=0)
args = parser.parse_args()



kernel_time = args.t / args.accesses
free_time = kernel_time * (1 - args.frac)
lock_time = kernel_time * (args.frac)

@ray.remote(num_gpus=args.workers, max_calls=0)
def waste_time(i, deps):
    if args.verbose:
        inner_start_t = time.perf_counter()
        print("Task", i, " | Start", flush=True)

    for k in range(args.accesses):
        sleep(lock_time)
        # free_sleep(free_time)
        # lock_sleep(lock_time)

    if args.verbose:
        inner_end_t = time.perf_counter()
        print("Task", i, " | Inner Time: ",
              inner_end_t - inner_start_t, flush=True)
        

def main():
    ray.init(num_gpus=4, runtime_env={"pip": ["cupy-cuda11x"]})

    start_t = time.perf_counter()
    n = args.b
    gemm1 = {}
    for j in range(n):
        for k in range(j):
            syrk_deps = [f'syrk_{j}_{_}' for _ in range(k)]
            syrk_deps += [f'solve_{j}_{k}']
            gemm1[f'syrk_{j}_{k}'] = waste_time.remote((i, j), syrk_deps)

        potrf_deps = [f'syrk_{j}_{_}' for _ in range(j)]
        ready_futures, _ = ray.wait(potrf_deps, num_returns=len(potrf_deps))
        #futures[f"barrier_{i}"] = ready_futures
        gemm1[f'potrf_{j}'] = waste_time.remote(j, ready_futures)
    
        for i in range(j+1, n):
            for k in range(j):
                gemm_deps = [f'solve_{j}_{k}', f'solve_{i}_{k}']
                gemm_deps += [f'gemm_{i}_{j}_{_}' for _ in range(k)]
                gemm1[f'gemm_{i}_{j}_{k}'] = waste_time.remote((i, j, k), gemm_deps)

            solve_deps = [f'gemm_{i}_{j}_{_}' for _ in range(j)]
            solve_deps += [f'potrf_{j}']
            ready_futures, _ = ray.wait(potrf_deps, num_returns=len(solve_deps))
            gemm1[f'solve_{i}_{j}'] = waste_time.remote((i, j), ready_futures)
    
    output = ray.get(n - 1)

    end_t = time.perf_counter()
    elapsed_t = end_t - start_t

    print(', '.join([str(args.workers), str(args.b), str(args.t),
    str(args.accesses), str(args.frac), str(elapsed_t)]), flush=True)

    ray.shutdown()

main()