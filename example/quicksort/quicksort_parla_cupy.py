import numpy as np
import cupy as cp

np.random.seed(10)
cp.random.seed(10)

import time

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace as TaskSpace
from parla.cython.device_manager import gpu

# COMMENT(wlr): Our distributed array here will be a list of cupy arrays. Everything will be managed manually

from common import create_array, get_size_info, balance_partition
from common_parla import partition, scatter


def quicksort(
        idx, global_prefix, global_A, global_workspace, start, end, T):
    
    start = int(start)
    end = int(end)

    # # How to select the active block from the global array
    global_start_idx = np.searchsorted(global_prefix, start, side="right") - 1
    global_end_idx = np.searchsorted(global_prefix, end, side="right") - 1

    # Split within a block at the endpoints (to form slices)
    local_left_split = start - global_prefix[global_start_idx]
    local_right_split = end - global_prefix[global_end_idx]

    local_left_split = (int)(local_left_split)
    local_right_split = (int)(local_right_split)

    A = []
    workspace = []
    
    #Reform a global array out of sliced components (NOTE: THESE SEMANTICS BREAK PARRAY. Concurrent slices cannot be written)
    if global_start_idx == global_end_idx and local_left_split < local_right_split:
        A.append(global_A[global_start_idx][local_left_split:local_right_split])
        workspace.append(global_workspace[global_start_idx])
    else:
        if (global_start_idx < global_end_idx) and (
            local_left_split < len(global_A[global_start_idx])
        ):
            A.append(global_A[global_start_idx][local_left_split:])
            workspace.append(global_workspace[global_start_idx][local_left_split:])

        for i in range(global_start_idx + 1, global_end_idx):
            A.append(global_A[i])
            workspace.append(global_workspace[i])

        if (global_end_idx < len(global_A)) and local_right_split > 0:
            A.append(global_A[global_end_idx][:local_right_split])
            workspace.append(global_workspace[global_end_idx][:local_right_split])


    n_partitions = len(A)
    device_list = tuple([gpu(arr.device.id) for arr in A])

    @spawn(T[idx], placement=[device_list], vcus=1)
    def quicksort_task():
        nonlocal global_prefix
        nonlocal global_A
        nonlocal global_workspace
        nonlocal start
        nonlocal end
        nonlocal T

        n_partitions = len(A)

        #print("TASK", T[idx], get_current_context(), flush=True)

        if n_partitions <= 1:
            # Base case.
            if n_partitions == 1:
                # The data is on a single gpu.
                A[0].sort()
            return

        #Form local prefix sum
        sizes = np.zeros(len(A) + 1, dtype=np.uint32)
        for i in range(len(A)):
            sizes[i + 1] = len(A[i])

        #Choose random pivot
        pivot_block = np.random.randint(0, n_partitions)
        pivot_idx = np.random.randint(0, len(A[pivot_block]))
        pivot = (int)(A[pivot_block][pivot_idx])

        # Perform local partition and repacking (no communication)
        left_counts = partition(A, workspace, pivot)
        local_left_count = np.sum(left_counts)
        global_left_count = start + local_left_count

        # compute communication pattern
        left_info, right_info = balance_partition(A, left_counts)

        # Send left points to left partition and right points to right partition (communication)
        scatter(A, workspace, left_info, right_info)

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
    # Per device size. 

    #This is also treated as the maximum number of points that can be on each device (very strict constraint).
    #This has a large performance impact due to recursion on the boundaries between partitions.
    #To do: Separate this into two variables?

    global_array, A, workspace = create_array(args.m, args.num_gpus)

    sizes, size_prefix = get_size_info(A)



    with Parla():
        T = TaskSpace("T")
        t_start = time.perf_counter()
        quicksort(1, size_prefix, A, workspace, 0, args.m * args.num_gpus, T)
        T.wait()
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
