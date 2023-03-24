
# from parla import Parla, spawn, TaskSpace
import argparse
import cupy as cp
import numpy as np
import crosspy as xp
from crosspy import cpu, gpu

from parla import parray as pa

from numba import njit

parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
parser.add_argument("-num_gpus", type=int, default=3)
args = parser.parse_args()

# np.random.seed(10)
# cp.random.seed(10)

# COMMENT(wlr): Our distributed array here will be a list of Parla arrays


def partition_kernel(A, B, comp, pivot):
    # TODO(wlr): Fuse this kernel
    bufferA = A.array
    bufferB = B.array
    comp[:] = (bufferA <= pivot)
    mid = comp.sum()
    bufferB[:mid] = bufferA[comp]
    bufferB[mid:] = bufferA[~comp]
    return mid


def partition(A, B, pivot):
    n_partitions = len(A)
    mid = np.zeros(n_partitions, dtype=np.uint32)

    for i, (array_in, array_out) in enumerate(zip(A, B)):
        with cp.cuda.Device(0):
            comp = cp.empty_like(array_in, dtype=cp.bool_)
            mid[i] = partition_kernel(array_in, array_out, comp, pivot)
    return mid


def balance_partition(A, left_counts):

    sizes = np.zeros(len(A), dtype=np.uint32)
    for i, array in enumerate(A):
        sizes[i] = len(array)

    remaining_left = np.copy(left_counts)
    remaining_right = np.copy(sizes) - left_counts
    free = np.copy(sizes)

    source_start_left = np.zeros((len(A), len(A)), dtype=np.uint32)
    target_start_left = np.zeros((len(A), len(A)), dtype=np.uint32)
    sz_left = np.zeros((len(A), len(A)), dtype=np.uint32)

    source_start_right = np.zeros((len(A), len(A)), dtype=np.uint32)
    target_start_right = np.zeros((len(A), len(A)), dtype=np.uint32)
    sz_right = np.zeros((len(A), len(A)), dtype=np.uint32)

    # Pack all left data to the left first
    target_idx = 0
    local_target_start = 0

    for source_idx in range(len(A)):
        local_source_start = 0
        message_size = remaining_left[source_idx]
        while (message_size > 0):
            max_message = min(free[target_idx], message_size)

            if max_message == 0:
                target_idx += 1
                local_target_start = 0
                continue

            free[target_idx] -= max_message
            remaining_left[source_idx] -= max_message

            sz_left[source_idx, target_idx] = max_message
            source_start_left[source_idx, target_idx] = local_source_start
            target_start_left[source_idx, target_idx] = local_target_start
            local_source_start += max_message
            local_target_start += max_message

            message_size = remaining_left[source_idx]

    # Pack all right data to the right
    for source_idx in range(len(A)):
        local_source_start = left_counts[source_idx]
        message_size = remaining_right[source_idx]
        while (message_size > 0):
            max_message = min(free[target_idx], message_size)

            if max_message == 0:
                target_idx += 1
                local_target_start = 0
                continue

            free[target_idx] -= max_message
            remaining_right[source_idx] -= max_message

            sz_right[source_idx, target_idx] = max_message
            source_start_right[source_idx, target_idx] = local_source_start
            target_start_right[source_idx, target_idx] = local_target_start
            local_source_start += max_message
            local_target_start += max_message

            message_size = remaining_right[source_idx]

    return (source_start_left, target_start_left, sz_left), (source_start_right, target_start_right, sz_right)


def scatter(A, B, left_info, right_info):

    source_starts, target_starts, sizes = left_info

    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with cp.cuda.Device(0):
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                # print(source_idx, target_idx, (source_start,
                #      source_end), (target_start, target_end))

                A[target_idx].array[target_start:target_end] = B[source_idx].array[source_start:source_end]

    source_starts, target_starts, sizes = right_info
    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with cp.cuda.Device(0):
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                # print(source_idx, target_idx, (source_start,
                #      source_end), (target_start, target_end))

                A[target_idx].array[target_start:target_end] = B[source_idx].array[source_start:source_end]


def quicksort(A, workspace, T):

    if len(A) == 1:
        A[0].array.sort()

    n_partitions = len(A)

    pivot = (int)(A[-1][-1])

    print(pivot)

    # local partition
    left_counts = partition(A, workspace, pivot)
    global_left_count = np.sum(left_counts)

    # compute communication pattern
    left_info, right_info = balance_partition(A, left_counts)

    # Send left to left and right to right
    scatter(A, workspace, left_info, right_info)

    sizes = np.zeros(len(A)+1, dtype=np.uint32)
    for i in range(len(A)):
        sizes[i+1] = len(A[i])
    size_prefix = np.cumsum(sizes)

    split_idx = np.searchsorted(size_prefix, global_left_count, side="right")-1
    local_split = global_left_count - size_prefix[split_idx]

    # print(local_split)
    print(pivot, global_left_count, split_idx)

    array_left = []
    array_left += [A[i] for i in range(split_idx)]
    array_left += [A[split_idx][0:local_split]]

    workspace_left = []
    workspace_left += [workspace[i] for i in range(split_idx)]
    workspace_left += [workspace[split_idx][0:local_split]]

    array_right = []
    array_right += [A[split_idx][local_split:sizes[split_idx+1]]]
    array_right += [A[i] for i in range(split_idx+1, len(A))]

    workspace_right = []
    workspace_right += [workspace[split_idx][local_split:sizes[split_idx+1]]]
    workspace_right += [workspace[i] for i in range(split_idx+1, len(A))]

    print("Left")
    for array in array_left:
        print(array)

    print("Right")
    for array in array_right:
        print(array)

    quicksort(array_left, workspace_left, T)
    quicksort(array_right, workspace_right, T)

    # Scatter to other partitions
    # scatter(active_A, active_B, mid)


def main(T):

    # Per device size
    m = 5

    # Initilize a CrossPy Array
    cupy_list_A = []
    cupy_list_B = []
    for i in range(args.num_gpus):
        with cp.cuda.Device(0) as dev:
            random_array = cp.random.randint(0, 100, size=m)
            random_array = random_array.astype(cp.int32)

            cupy_list_A.append(random_array)
            cupy_list_B.append(cp.zeros(m, dtype=cp.int32))
            dev.synchronize()

    A = pa.asarray_batch(cupy_list_A)

    for array in A:
        print(array)

    workspace = pa.asarray_batch(cupy_list_B)

    quicksort(A, workspace, T)


if __name__ == "__main__":
    T = None
    main(T)
