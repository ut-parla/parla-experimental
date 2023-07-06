import numpy as np
import cupy as cp

np.random.seed(10)
cp.random.seed(10)

import time

# COMMENT(wlr): Our distributed array here will be a list of Parla arrays

from common import create_array, get_size_info, balance_partition
from common import partition, scatter


def quicksort(
        global_prefix, A, workspace, start, end):
    n_partitions = len(A)
    print("LENGTH", n_partitions, end - start)

    sizes = np.zeros(len(A)+1, dtype=np.uint32)
    for i in range(len(A)):
        # print("INCOMING ARRAY", A[i].array)
        # sizes[i+1] = len(A[i].array)
        print("INCOMING ARRAY", len(A[i]))
        sizes[i+1] = len(A[i])

    size_prefix = np.cumsum(sizes)
    local_size = np.sum(sizes)

    # print("SIZES", size_prefix)

    if len(A) == 1:
        # print("BASE")
        # A[0].array.sort()
        A[0].sort()
        return

    if len(A) == 0:
        return

    n_partitions = len(A)

    # Choose random pivot
    pivot_block = np.random.randint(0, n_partitions)
    pivot_idx = np.random.randint(0, len(A[pivot_block]))
    pivot = (int)(A[pivot_block][pivot_idx])

    # print("Pivot: ", pivot)

    # local partition
    left_counts = partition(A, workspace, pivot)
    global_left_count = np.sum(left_counts)

    # compute communication pattern
    left_info, right_info = balance_partition(A, left_counts)

    # Send left to left and right to right
    scatter(A, workspace, left_info, right_info)

    # print(size_prefix, global_left_count)
    split_idx = np.searchsorted(size_prefix, global_left_count, side="right")-1
    local_split = global_left_count - size_prefix[split_idx]

    # print(local_split)
    # print("Split Index: ", split_idx)

    array_left = []
    array_left += [A[i] for i in range(split_idx)]
    if split_idx < len(A) and local_split > 0:
        array_left += [A[split_idx][0:local_split]]

    workspace_left = []
    workspace_left += [workspace[i] for i in range(split_idx)]
    if split_idx < len(A) and local_split > 0:
        workspace_left += [workspace[split_idx][0:local_split]]

    array_right = []
    if split_idx < len(A) and local_split < sizes[split_idx+1]:
        array_right += [A[split_idx][local_split:sizes[split_idx+1]]]
    array_right += [A[i] for i in range(split_idx+1, len(A))]

    workspace_right = []
    if split_idx < len(A) and local_split < sizes[split_idx+1]:
        workspace_right += [workspace[split_idx]
                            [local_split:sizes[split_idx+1]]]
    workspace_right += [workspace[i] for i in range(split_idx+1, len(A))]

    # print("Left", len(array_left), global_left_count)
    # for array in array_left:
    # print(array.array)
    #    print(array)

    # print("Right", len(array_right), local_size - global_left_count)
    # for array in array_right:
    # print(array.array)
    #    print(array)

    quicksort(global_prefix, array_left, workspace_left,
              start, start+global_left_count)
    quicksort(global_prefix, array_right, workspace_right,
              start+global_left_count, end)

    # Scatter to other partitions
    # scatter(active_A, active_B, mid)


def main(args):

    global_array, A, workspace = create_array(args.m, args.num_gpus)

    sizes, size_prefix = get_size_info(A)

    t_start = time.perf_counter()
    with cp.cuda.Device(0) as d:
        quicksort(size_prefix, A, workspace, 0, args.m * args.num_gpus)
        d.synchronize()
    t_end = time.perf_counter()

    print("Time: ", t_end - t_start)

    with cp.cuda.Device(0) as d:
        d.synchronize()
        t_start = time.perf_counter()
        global_array.sort()
        d.synchronize()
    t_end = time.perf_counter()

    print("Time: ", t_end - t_start)

    print("Sorted")
    for array in A:
        print(array)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
    parser.add_argument("-num_gpus", type=int, default=2)
    parser.add_argument("-m", type=int, default=10)
    args = parser.parse_args()
    main(args)
