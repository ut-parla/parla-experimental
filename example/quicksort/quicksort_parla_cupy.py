# from parla import Parla, spawn, TaskSpace
import argparse
import cupy as cp
import numpy as np
import time

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace as TaskSpace
from parla.cython.device_manager import gpu
from parla.common.globals import get_current_context
from parla.common.array import copy

parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
parser.add_argument("-num_gpus", type=int, default=4)
parser.add_argument("-m", type=int, default=10000)
args = parser.parse_args()


#Note: this experimental allocator breaks memcpy async
#cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)

np.random.seed(10)
cp.random.seed(10)

# COMMENT(wlr): Our distributed array here will be a list of cupy arrays. Everything will be managed manually


def partition_kernel(A, B, comp, pivot):
    # TODO(wlr): Fuse this kernel
    bufferA = A
    bufferB = B
    comp[:] = bufferA[:] < pivot
    mid = (int)(comp.sum())
    bufferB[:mid] = bufferA[comp]
    bufferB[mid:] = bufferA[~comp]
    return mid


def partition(A, B, pivot):
    context = get_current_context()
    n_partitions = len(A)
    mid = np.zeros(n_partitions, dtype=np.uint32)

    for i, (array_in, array_out) in enumerate(zip(A, B)):
        with context.devices[i]:
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

            with context.devices[target_idx]:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                target = A[target_idx]
                source = B[source_idx]

                copy(
                    target[target_start:target_end], 
                    source[source_start:source_end]
                    )


    source_starts, target_starts, sizes = right_info
    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with context.devices[target_idx]:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                target = A[target_idx]
                source = B[source_idx]

                copy(
                    target[target_start:target_end], 
                    source[source_start:source_end]
                    )

                


def quicksort(idx, global_prefix, global_A, global_workspace, start, end, T):
    
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

        if n_partitions == 1:
            #Base case. The data is on a single gpu.
            A[0].sort()
            return

        if n_partitions == 0:
            #Base case. The array is empty.
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

def main():
    # Per device size. 

    #This is also treated as the maximum number of points that can be on each device (very strict constraint).
    #This has a large performance impact due to recursion on the boundaries between partitions.
    #To do: Separate this into two variables?

    m = args.m
    global_array = np.arange(m*args.num_gpus, dtype=np.int32)
    np.random.shuffle(global_array)

    # Init a distributed crosspy array
    cupy_list_A = []
    cupy_list_B = []
    for i in range(args.num_gpus):
        with cp.cuda.Device(i) as dev:
            random_array = cp.asarray(global_array[m*i:m*(i+1)])
            cupy_list_A.append(random_array)
            cupy_list_B.append(cp.empty(m, dtype=cp.int32))
    
    for i in range(args.num_gpus):
        with cp.cuda.Device(i) as dev:
            dev.synchronize()

    A = cupy_list_A
    workspace = cupy_list_B

    #Compute parition size prefix
    sizes = np.zeros(len(A) + 1, dtype=np.uint32)
    for i in range(len(A)):
        sizes[i + 1] = len(A[i])
    size_prefix = np.cumsum(sizes)


    with Parla():
        T = TaskSpace("T")
        t_start = time.perf_counter()
        quicksort(1, size_prefix, A, workspace, 0, m * args.num_gpus, T)
        T.wait()
        t_end = time.perf_counter()

    print("Time: ", t_end - t_start)

    print("Sorted")
    for array in A:
        print(array)


if __name__ == "__main__":
    main()
