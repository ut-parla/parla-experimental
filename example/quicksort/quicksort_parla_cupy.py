import numpy as np
import cupy as cp

np.random.seed(10)
cp.random.seed(10)

import time

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace as TaskSpace
from parla.cython.device_manager import gpu
from parla.common.globals import get_current_context
from parla.common.array import copy

from common import create_array, get_size_info
from common_parla import partition
from common_parla import parla_copy
# from cupy_helper import cupy_copy

from cupy_helper import slice_distributed_arrays, balance_partition, scatter

def quicksort(global_dist_arrays: list, global_workspace: list, global_prefix, start, end, T, Tid=1):
    A, workspace = slice_distributed_arrays(global_dist_arrays, global_prefix, start, end, workspace=global_workspace)

    placement = tuple([gpu(a.device.id) for a in A])
    # print("Spawning task with device list: ", placement, flush=True)
    @spawn(T[Tid], placement=[placement], vcus=1)
    def quicksort_task():
        nonlocal global_dist_arrays
        nonlocal global_workspace
        nonlocal global_prefix
        nonlocal start
        nonlocal end
        nonlocal T

        context = get_current_context()
        n_partitions = len(A)

        #print("TASK", T[idx], get_current_context(), flush=True)

        if n_partitions < 2:
            # Base case.
            if n_partitions == 1:
                # The data is on a single gpu.
                A[0].sort()
            return

        # Choose random pivot
        pivot_block = np.random.randint(0, n_partitions)
        pivot_idx = np.random.randint(0, len(A[pivot_block]))
        pivot = (int)(A[pivot_block][pivot_idx])

        # Perform local partition and repacking (no communication)
        mid = partition(A, pivot, workspace=workspace)
        left_count = np.sum(mid)

        # compute communication pattern
        left_info, right_info = balance_partition(A, mid)

        # Send left points to left partition and right points to right partition (communication)
        scatter(A, workspace, left_info, right_info, ctx_func=lambda i: context.devices[i], copy_func=parla_copy)

        quicksort(
            global_dist_arrays, global_workspace,
            global_prefix, start, start + left_count,
            T, 2 * Tid,
        )
        quicksort(
            global_dist_arrays, global_workspace,
            global_prefix, start + left_count, end,
            T, 2 * Tid + 1,
        )

def main(args):
    global_array, dist_array_list, workspace = create_array(args.m, args.num_gpus)

    sizes, size_prefix = get_size_info(dist_array_list)

    print("Original Array: ", dist_array_list, flush=True)

    with Parla():
        T = TaskSpace("T")
        t_start = time.perf_counter()
        quicksort(dist_array_list, workspace, size_prefix, 0, args.m * args.num_gpus, T)
        T.wait()  # await T
        t_end = time.perf_counter()

    print("Sorted:")
    for array in dist_array_list:
        print(array)
    print("Time:")
    print(t_end - t_start)

    with cp.cuda.Device(0) as d:
        d.synchronize()
        t_start = time.perf_counter()
        global_array.sort()
        d.synchronize()
        t_end = time.perf_counter()

    print("Time cupy.sort on single device:")
    print(t_end - t_start)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
    parser.add_argument("-num_gpus", type=int, default=2)
    parser.add_argument("-m", type=int, default=10)
    args = parser.parse_args()
    main(args)
