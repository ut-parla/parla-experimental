import os
import time
import ray
from concurrent.futures import ThreadPoolExecutor

# from parla import sleep_nogil, sleep_gil
# from parla import sleep_nogil as sleep
from parla import sleep_nogil as free_sleep
from parla import sleep_gil as lock_sleep
import argparse

os.environ["RAY_DEDUP_LOGS"] = str(0)

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=14)
parser.add_argument('-workers', default=4, type=int)
parser.add_argument('-t', type=int, default=1000)
parser.add_argument('-accesses', type=int, default=1)
parser.add_argument('-verbose', type=int, default=1)
parser.add_argument('-frac', type=float, default=1)
args = parser.parse_args()

gil_fraction = args.frac 
accesses = args.accesses 
sleep_time = args.t 

kernel_time = sleep_time / accesses
free_time = kernel_time * (1 - gil_fraction)
lock_time = kernel_time * gil_fraction

# @ray.remote(num_gpus=args.workers, max_calls=0)
# def waste_time(i, deps):
#     if args.verbose:
#         inner_start_t = time.perf_counter()
#         print("Task", i, " | Start", flush=True)

#     for k in range(args.accesses):
#         # sleep(lock_time)
#         free_sleep(free_time)
#         lock_sleep(lock_time)

#     if args.verbose:
#         inner_end_t = time.perf_counter()
#         print("Task", i, " | Inner Time: ",
#               inner_end_t - inner_start_t, flush=True)
        
@ray.remote(num_gpus=args.workers, max_calls=0)
def final(deps):
    return

def main():
    ray.init(num_gpus=4, runtime_env={"pip": ["cupy-cuda11x"]})
    
    @ray.remote(num_gpus=args.workers, max_calls=0)
    def waste_time(i, deps):
        if args.verbose:
            inner_start_t = time.perf_counter()
            print("Task", i, " | Start", flush=True)

        for k in range(args.accesses):
            # sleep(lock_time)
            free_sleep(free_time)
            lock_sleep(lock_time)

        if args.verbose:
            inner_end_t = time.perf_counter()
            print("Task", i, " | Inner Time: ",
                  inner_end_t - inner_start_t, flush=True)

    start_t = time.perf_counter()
    n = args.b
    gemm1 = {}
    for j in range(n):
        for k in range(j):
            syrk_deps = [f'syrk_{j}_{_}' for _ in range(k)]
            syrk_deps += [f'solve_{j}_{k}']
            gemm1[f'syrk_{j}_{k}'] = waste_time.remote((i, j), syrk_deps)
            #ray.get(gemm1[f'syrk_{j}_{k}'])

        potrf_deps = [f'syrk_{j}_{_}' for _ in range(j)]
        # ready_futures, _ = ray.wait(potrf_deps, num_returns=len(potrf_deps))
        #futures[f"barrier_{i}"] = ready_futures
        gemm1[f'potrf_{j}'] = waste_time.remote(j, potrf_deps)
        ray.get(gemm1[f'potrf_{j}'])
    
        for i in range(j+1, n):
            for k in range(j):
                gemm_deps = [f'solve_{j}_{k}', f'solve_{i}_{k}']
                gemm_deps += [f'gemm_{i}_{j}_{_}' for _ in range(k)]
                gemm1[f'gemm_{i}_{j}_{k}'] = waste_time.remote((i, j, k), gemm_deps)
                #ray.get(gemm1[f'gemm_{i}_{j}_{k}'])

            solve_deps = [f'gemm_{i}_{j}_{_}' for _ in range(j)]
            solve_deps += [f'potrf_{j}']
            # ready_futures, _ = ray.wait(solve_deps, num_returns=len(solve_deps))
            gemm1[f'solve_{i}_{j}'] = waste_time.remote((i, j), solve_deps)
            ray.get(gemm1[f'solve_{i}_{j}'])
    
    final_sync = final.remote(gemm1[f"potrf_{args.b-1}"])
    output = ray.get(final_sync)

    end_t = time.perf_counter()
    elapsed_t = end_t - start_t

    print(', '.join([str(args.workers), str(args.b), str(args.t),
    str(args.accesses), str(args.frac), str(elapsed_t)]), flush=True)

    # ray.shutdown()

main()