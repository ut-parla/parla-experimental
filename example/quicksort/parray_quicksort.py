# from parla import Parla, spawn, TaskSpace
import argparse
import cupy as cp
import numpy as np
import time

from parla import parray as pa
from parla import Parla, spawn, TaskSpace
from parla.cython.device_manager import cuda
from parla.common.globals import get_current_context

parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
parser.add_argument("-num_gpus", type=int, default=3)
args = parser.parse_args()

np.random.seed(10)
cp.random.seed(10)

# COMMENT(wlr): Our distributed array here will be a list of Parla arrays


def partition_kernel(A, B, comp, pivot):
    # TODO(wlr): Fuse this kernel
    bufferA = A.array
    bufferB = B.array
    comp[:] = bufferA[:] < pivot
    mid = (int)(comp.sum())
    bufferB[:mid] = bufferA[comp]
    bufferB[mid:] = bufferA[~comp]
    # print("Reordered Buffer:", bufferA, comp, bufferB, (bufferB[:] < pivot))
    return mid


def partition(A, B, pivot):
    context = get_current_context()
    n_partitions = len(A)
    mid = np.zeros(n_partitions, dtype=np.uint32)

    for i, (array_in, array_out) in enumerate(zip(A, B)):
        with context.device[0]:
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
        while message_size > 0:
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
        while message_size > 0:
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

    return (source_start_left, target_start_left, sz_left), (
        source_start_right,
        target_start_right,
        sz_right,
    )


def scatter(A, B, left_info, right_info):
    context = get_current_context()

    source_starts, target_starts, sizes = left_info

    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with context.device[0]:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                # print(source_idx, target_idx, (source_start,
                #      source_end), (target_start, target_end))

                target = A[target_idx]
                source = B[source_idx]

                print("Target:", target, flush=True)
                print("Source: ", source, flush=True)

                A[target_idx].array[target_start:target_end] = cp.asarray(B[source_idx].array[
                    source_start:source_end
                ])
                # print("TARGET: ", target, type(target))
                # print("SOURCE: ", source, type(source))
                # target[target_start:target_end] = source[source_start:source_end]

    source_starts, target_starts, sizes = right_info
    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with context.device[0]:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                # print(source_idx, target_idx, (source_start,
                #      source_end), (target_start, target_end))

                A[target_idx].array[target_start:target_end] = cp.asarray(B[source_idx].array[
                    source_start:source_end
                ])
                # target = A[target_idx]
                # source = B[source_idx]
                # target[target_start:target_end] = source[source_start:source_end]


def quicksort(idx, global_prefix, global_A, global_workspace, start, end, T):

    start = int(start)
    end = int(end)

    # # How to select the active block from the global array
    global_start_idx = np.searchsorted(global_prefix, start, side="right") - 1
    global_end_idx = np.searchsorted(global_prefix, end, side="right") - 1

    # Split within a blocks at the endpoints (form slices)
    local_left_split = start - global_prefix[global_start_idx]
    local_right_split = end - global_prefix[global_end_idx]

    local_left_split = (int)(local_left_split)
    local_right_split = (int)(local_right_split)

    A = []
    workspace = []

    print("GLOBAL_PREFIX: ", global_prefix)
    print("START: ", start, "END: ", end)
    print("GLOBAL_START: ", global_start_idx, "GLOBAL_END: ", global_end_idx)
    print("LOCAL_START: ", local_left_split, "LOCAL_END: ", local_right_split)

    if global_start_idx == global_end_idx and local_left_split < local_right_split:
        A.append(global_A[global_start_idx]
                 [local_left_split:local_right_split])
        workspace.append(global_workspace[global_start_idx])
    else:
        print(
            "local_left_split: ",
            local_left_split,
            "len: ",
            len(global_A[global_start_idx]),
        )
        if (global_start_idx < global_end_idx) and (
            local_left_split < len(global_A[global_start_idx])
        ):
            A.append(global_A[global_start_idx][local_left_split:])
            workspace.append(
                global_workspace[global_start_idx][local_left_split:])
            print("ADDED LEFT")

        for i in range(global_start_idx + 1, global_end_idx):
            A.append(global_A[i])
            workspace.append(global_workspace[i])
            print("ADDED MIDDLE")

        if (global_end_idx < len(global_A)) and local_right_split > 0:
            A.append(global_A[global_end_idx][:local_right_split])
            workspace.append(
                global_workspace[global_end_idx][:local_right_split])
            print("ADDED RIGHT")

    n_partitions = len(A)

    tag_A = [(A[i], i) for i in range(len(A))]
    tag_B = [(workspace[i], i) for i in range(len(workspace))]

    unpacked = tag_A + tag_B

    print("unpacked: ", unpacked)

# TODO(hc): just cuda doesnt work well.
    device_list = tuple([cuda(i) for i in range(n_partitions)])
# device_list = cuda

    @spawn(T[idx], placement=[device_list], inout=unpacked)
    def quicksort_task():
        nonlocal global_prefix
        nonlocal global_A
        nonlocal global_workspace
        nonlocal start
        nonlocal end
        nonlocal T
        n_partitions = len(A)

        print("TASK", T[idx], flush=True)

        print("LENGTH", n_partitions, end - start, start, end)

        sizes = np.zeros(len(A) + 1, dtype=np.uint32)
        for i in range(len(A)):
            # print("INCOMING ARRAY", A[i].array)
            sizes[i + 1] = len(A[i].array)
            # print("INCOMING ARRAY", A[i], len(A[i]))
            # sizes[i+1] = len(A[i])

        np.cumsum(sizes)
        np.sum(sizes)

        if len(A) == 1:
            # print("BASE")
            A[0].array.sort()
            # A[0].sort()
            return

        if len(A) == 0:
            return

        n_partitions = len(A)

        # # Choose random pivot
        pivot_block = np.random.randint(0, n_partitions)
        pivot_idx = np.random.randint(0, len(A[pivot_block]))
        pivot = (int)(A[pivot_block][pivot_idx])

        print("Pivot: ", pivot)

        # local partition
        left_counts = partition(A, workspace, pivot)
        local_left_count = np.sum(left_counts)
        global_left_count = start + local_left_count

        print("LOCAL LEFT COUNT: ", local_left_count)

        # compute communication pattern
        left_info, right_info = balance_partition(A, left_counts)

        # Send left to left and right to right
        scatter(A, workspace, left_info, right_info)

        print("-------")

        quicksort(
            2 * idx,
            global_prefix,
            global_A,
            global_workspace,
            start,
            global_left_count,
            T,
        )
        quicksort(
            2 * idx + 1,
            global_prefix,
            global_A,
            global_workspace,
            global_left_count,
            end,
            T,
        )

    # Scatter to other partitions
    # scatter(active_A, active_B, mid)


def main():
    # Per device size
    m = 10

    # Initilize a CrossPy Array
    cupy_list_A = []
    cupy_list_B = []
    for i in range(args.num_gpus):
        with cp.cuda.Device(0) as dev:
            random_array = cp.random.randint(0, 1000000, size=m)
            random_array = random_array.astype(cp.int32)

            cupy_list_A.append(random_array)
            cupy_list_B.append(cp.zeros(m, dtype=cp.int32))
            dev.synchronize()

    A = pa.asarray_batch(cupy_list_A)
    # A = cupy_list_A

    sizes = np.zeros(len(A) + 1, dtype=np.uint32)
    for i in range(len(A)):
        sizes[i + 1] = len(A[i])
    size_prefix = np.cumsum(sizes)

    workspace = pa.asarray_batch(cupy_list_B)
    # workspace = cupy_list_B

    with Parla():
        T = TaskSpace("T")
        t_start = time.perf_counter()
        with cp.cuda.Device(0) as d:
            quicksort(1, size_prefix, A, workspace, 0, m * args.num_gpus, T)
    t_end = time.perf_counter()

    print("Time: ", t_end - t_start)

    with cp.cuda.Device(0) as d:
        test = cp.random.randint(0, 100000000000, size=m * args.num_gpus)
        test = test.astype(cp.int32)
        d.synchronize()
        t_start = time.perf_counter()
        test.sort()
        d.synchronize()
    t_end = time.perf_counter()

    print("Time: ", t_end - t_start)

    print("Sorted")
    for array in A:
        print(array)


if __name__ == "__main__":
    main()
