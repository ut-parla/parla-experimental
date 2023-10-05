import time
import ray
import argparse
#from parla import sleep_nogil as free_sleep
#from parla import sleep_gil as lock_sleep
from doozer.cpu import sleep as free_sleep
from doozer.cpu import sleep_with_gil as lock_sleep
# Initialize Ray

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=4)
parser.add_argument('-workers', default=4, type=int)
parser.add_argument('-t', type=int, default=1000)
parser.add_argument('-accesses', type=int, default=1)
parser.add_argument('-verbose', type=int, default=1)
parser.add_argument('-frac', type=float, default=0)
args = parser.parse_args()

gil_fraction = args.frac 
accesses = args.accesses 
sleep_time = args.t 

kernel_time = sleep_time / accesses
free_time = kernel_time * (1 - gil_fraction)
lock_time = kernel_time * gil_fraction


@ray.remote(num_cpus=1)
def waste_time(name, i, deps):
    if deps is not None:
        ray.get(deps)

    for k in range(args.accesses):
        #time.sleep(1)
        free_sleep(free_time)
        lock_sleep(lock_time)

    if args.verbose:
        print(name, " Task", i, " | END", flush=True)

if __name__ == '__main__':

    ray.init(num_cpus=4)
    #ray.init(num_gpus=4, runtime_env={"pip": ["cupy-cuda11x"]})

    start_t = time.perf_counter()
    n=args.b
    dsk = {}
    for j in range(n):
        for k in range(j):
            syrk_deps = [dsk[f'syrk_{j}_{_}'] for _ in range(k)]
            syrk_deps += [dsk[f'solve_{j}_{k}']]
            dsk[f'syrk_{j}_{k}'] = waste_time.remote(f'syrk_{j}_{k}', (j, k), syrk_deps)

        potrf_deps = [dsk[f'syrk_{j}_{_}'] for _ in range(j)]
        dsk[f'potrf_{j}'] = waste_time.remote(f'potrf_{j}', j, potrf_deps)

        for i in range(j + 1, n):
            for k in range(j):
                gemm_deps = [dsk[f'solve_{j}_{k}'], dsk[f'solve_{i}_{k}']]
                gemm_deps += [dsk[f'gemm_{i}_{j}_{_}'] for _ in range(k)]
                dsk[f'gemm_{i}_{j}_{k}'] = waste_time.remote(f'gemm_{i}_{j}_{k}', (i, j, k), gemm_deps)

            solve_deps = [dsk[f'gemm_{i}_{j}_{_}'] for _ in range(j)]
            solve_deps += [dsk[f'potrf_{j}']]
            dsk[f'solve_{i}_{j}'] = waste_time.remote(f'solve_{i}_{j}', (i, j), solve_deps)
    
    ray.get(dsk[f"potrf_{args.b-1}"])
    
    end_t = time.perf_counter()
    elapsed_t = end_t - start_t

    print(', '.join(["start",str(args.workers), str(args.b), str(args.t),
                     str(args.accesses), str(args.frac), str(elapsed_t)]), flush=True)
    ray.shutdown()