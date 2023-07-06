import numpy as np
import cupy as cp

np.random.seed(10)
cp.random.seed(10)

import time

from parla import Parla, spawn, TaskSpace
from parla.cython.device_manager import gpu as cuda

# COMMENT(wlr): Our distributed array here will be a list of Parla arrays

from common import create_array, get_size_info, balance_partition
from common_parla_manual import partition, scatter


def quicksort(
        idx, global_prefix, global_A, global_workspace, start, end, T):
    
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
        A.append(global_A[global_start_idx][local_left_split:local_right_split])
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
            workspace.append(global_workspace[global_start_idx][local_left_split:])
            print("ADDED LEFT")

        for i in range(global_start_idx + 1, global_end_idx):
            A.append(global_A[i])
            workspace.append(global_workspace[i])
            print("ADDED MIDDLE")

        if (global_end_idx < len(global_A)) and local_right_split > 0:
            A.append(global_A[global_end_idx][:local_right_split])
            workspace.append(global_workspace[global_end_idx][:local_right_split])
            print("ADDED RIGHT")


    tag_A = [(A[i], i) for i in range(len(A))]
    tag_B = [(workspace[i], i) for i in range(len(A))]

    unpacked = tag_A + tag_B

    print("unpacked: ", unpacked)

    device_list = tuple([cuda(a.device.id) for a in A])
    print("Spawning task with device list: ", device_list, flush=True)
    # device_list = cuda

    @spawn(T[idx], placement=[device_list])
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
            sizes[i + 1] = len(A[i])
            # print("INCOMING ARRAY", A[i], len(A[i]))
            # sizes[i+1] = len(A[i])


        if n_partitions <= 1:
            # Base case.
            if n_partitions == 1:
                # The data is on a single gpu.
                A[0].sort()
            return


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

def main(args):
    global_array, A, workspace = create_array(args.m, args.num_gpus)

    sizes, size_prefix = get_size_info(A)


    print("Original Array: ", A, flush=True)

    with Parla():
        T = TaskSpace("T")
        t_start = time.perf_counter()
        with cp.cuda.Device(0) as d:
            quicksort(1, size_prefix, A, workspace, 0, args.m * args.num_gpus, T)
    t_end = time.perf_counter()

    print("Time: ", t_end - t_start)

    with cp.cuda.Device(0) as d:
        test = cp.random.randint(0, 100000000000, size=args.m * args.num_gpus)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
    parser.add_argument("-num_gpus", type=int, default=2)
    parser.add_argument("-m", type=int, default=10)
    args = parser.parse_args()
    main(args)
