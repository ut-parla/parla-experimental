import numpy as np
import cupy as cp

np.random.seed(10)
cp.random.seed(10)

import time

from common import create_array, get_size_info
from common import partition, scatter

from cupy_helper import slice_distributed_arrays, balance_partition

def quicksort(dist_arrays: list, workspace: list):
    n_partitions = len(dist_arrays)

    # print("TASK", T[idx], get_current_context(), flush=True)

    if n_partitions < 2:
        # Base case.
        if n_partitions == 1:
            # The data is on a single gpu.
            dist_arrays[0].sort()
        return

    # Choose random pivot
    pivot_block = np.random.randint(0, n_partitions)
    pivot_idx = np.random.randint(0, len(dist_arrays[pivot_block]))
    pivot = (int)(dist_arrays[pivot_block][pivot_idx])

    # Perform local partition and repacking (no communication)
    mid = partition(dist_arrays, pivot, workspace=workspace)
    left_count = np.sum(mid)

    # compute communication pattern
    left_info, right_info = balance_partition(dist_arrays, mid)

    # Send left points to left partition and right points to right partition (communication)
    scatter(dist_arrays, workspace, left_info, right_info)

    sizes, size_prefix = get_size_info(dist_arrays)
    array_left, workspace_left = slice_distributed_arrays(dist_arrays, size_prefix, end=left_count, workspace=workspace)
    quicksort(array_left, workspace_left)
    array_right, workspace_right = slice_distributed_arrays(dist_arrays, size_prefix, start=left_count, workspace=workspace)
    quicksort(array_right, workspace_right)

def main(args):
    global_array, dist_array_list, workspace = create_array(args.m, args.num_gpus)

    t_start = time.perf_counter()
    quicksort(dist_array_list, workspace)
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
