import time
import ray
import argparse
from parla import sleep_nogil as free_sleep
from parla import sleep_gil as lock_sleep
# Initialize Ray

ray.init(num_cpus=4)
#ray.init(num_gpus=4, runtime_env={"pip": ["cupy-cuda11x"]})

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=14)
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

# free_sleep = ray.remote
# lock_sleep = ray.remote

@ray.remote(num_cpus=1)
def free_sleep_ray(duration):
    # free_sleep(duration) 
    time.sleep(duration / 1000)  # Sleep in seconds

@ray.remote(num_cpus=1)
def lock_sleep_ray(duration):
    # lock_sleep(duration)
    for _ in range(duration):
        pass  

@ray.remote(num_cpus=1)
def waste_time(i, deps):
    if args.verbose:
        inner_start_t = time.perf_counter()
        print("Task", i, " | Start", flush=True)

    for k in range(args.accesses):
        free_sleep_ray.remote(free_time)
        lock_sleep_ray.remote(lock_time)

    if args.verbose:
        inner_end_t = time.perf_counter()
        print("Task", i, " | Inner Time: ",
              inner_end_t - inner_start_t, flush=True)

@ray.remote(num_cpus=1)
def final(deps):
    return

@ray.remote(num_cpus=1)
def create_graph(n):
    dsk = {}
    for j in range(n):
        for k in range(j):
            syrk_deps = [f'syrk_{j}_{_}' for _ in range(k)]
            syrk_deps += [f'solve_{j}_{k}']
            dsk[f'syrk_{j}_{k}'] = waste_time.remote((j, k), syrk_deps)

        potrf_deps = [f'syrk_{j}_{_}' for _ in range(j)]
        dsk[f'potrf_{j}'] = waste_time.remote(j, potrf_deps)

        for i in range(j + 1, n):
            for k in range(j):
                gemm_deps = [f'solve_{j}_{k}', f'solve_{i}_{k}']
                gemm_deps += [f'gemm_{i}_{j}_{_}' for _ in range(k)]
                dsk[f'gemm_{i}_{j}_{k}'] = waste_time.remote((i, j, k), gemm_deps)

            solve_deps = [f'gemm_{i}_{j}_{_}' for _ in range(j)]
            solve_deps += [f'potrf_{j}']
            dsk[f'solve_{i}_{j}'] = waste_time.remote((i, j), solve_deps)
    
    # final_sync = final.remote(dsk[f"potrf_{args.b-1}"])
    # output = ray.get(final_sync)
    end_t = time.perf_counter()
    elapsed_t = end_t - start_t

    print(', '.join([str(args.workers), str(args.b), str(args.t),
    str(args.accesses), str(args.frac), str(elapsed_t)]), flush=True)
    return dsk

if __name__ == '__main__':
    start_t = time.perf_counter()
    
    dsk = create_graph.remote(args.b)
    
    # final_sync = final.remote(dsk[f"potrf_{args.b-1}"])
    output = ray.get(dsk)
    
    end_t = time.perf_counter()
    elapsed_t = end_t - start_t

    print(', '.join(["start",str(args.workers), str(args.b), str(args.t),
                     str(args.accesses), str(args.frac), str(elapsed_t)]), flush=True)
    ray.shutdown()