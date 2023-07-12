from parla import Parla, spawn, TaskSpace
from parla.common.globals import get_current_context
from parla.cython.device_manager import gpu
from parla import parray

import argparse
import cupy as cp
import numpy as np
import crosspy as xp
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("-ngpus", type=int, default=2)
parser.add_argument("-m", type=int, default=3)
args = parser.parse_args()

parla_wrapper = parray.asarray

np.random.seed(10)
cp.random.seed(10)

ngpus = args.ngpus


from common import partition_kernel



def partition(xA, pivot):

    context = get_current_context()

    print("Context in partition function: ", context)

    if isinstance(xA, cp.ndarray):
        n_partitions = 1
    else:
        n_partitions = len(xA.block_view())

    mid = np.zeros(n_partitions+1, dtype=np.uint32)

    for i in range(n_partitions):
        with context.devices[i]:
            print("Working on partition: ", i, cp.cuda.runtime.getDevice(), flush=True)
            if isinstance(xA, cp.ndarray):
                local_array = xA
            else:
                local_array = xA.values()[i].array
            workspace = cp.empty_like(local_array)
            comp = cp.empty_like(local_array, dtype=cp.bool_)
            mid[i+1] = partition_kernel(local_array, workspace, comp, pivot)
            local_array[:] = workspace[:]
    return mid


# TODO(wlr): Fuse this kernel, pack this better?
def scatter(splits, active_array, left_array, right_array):
    """
    :param xA, xB,  sliced crosspy arrays for the active element set
    :param splits, a host ndarray of how many left elements are in each partition 
    """

    n_partitions = len(active_array.values())

    sizes = np.zeros(n_partitions+1, dtype=np.uint32)
    for i, array in enumerate(active_array.values()):
        sizes[i+1] = len(array)

    size_prefix = np.cumsum(sizes)
    left_prefix = np.cumsum(splits)
    if left_array is not None:

        left_count = len(left_array)
        left_send_indices = np.zeros(left_count, dtype=np.uint32)

        for i in range(n_partitions):
            left_send_indices[left_prefix[i]:left_prefix[i+1]
                              ] = np.arange(size_prefix[i], size_prefix[i]+splits[i+1])

        # Performing left scatter
        # print("Left send indices: ", left_send_indices)
        # print("Right send indices: ", right_send_indices)
        # print("Source len: ", len(active_array), "Index Len: ", len(left_send_indices), "Target Len: ", len(left_array))
        xp.alltoallv(active_array, left_send_indices, left_array)

    if right_array is not None:
        right_count = len(right_array)
        right_prefix = size_prefix - left_prefix
        right_send_indices = np.zeros(right_count, dtype=np.uint32)
        for i in range(n_partitions):
            right_send_indices[right_prefix[i]:right_prefix[i+1]
                               ] = np.arange(size_prefix[i]+splits[i+1], size_prefix[i+1])

        # Performing right scatter
        xp.alltoallv(active_array, right_send_indices, right_array)

    return None


def quicksort(idx, global_array, active_array, active_slice, T):

    n_partitions = len(active_array.values())

    if n_partitions == 1:
        #dev_id = active_array.values()[0][0].device.id
        #print("CREATING TASK CONSTRAINTS: ", idx, active_array, dev_id)
        placement = gpu[{'vcus': 1000}]
    else:
        placement = tuple(
                [gpu[{'vcus': 1000}] for arr in active_array.block_view()])

        print("CREATING TASK CONSTRAINTS: ", idx, active_array, [arr.device.id for arr in active_array.block_view()])

    print("Placement for : ", idx, placement, flush=True)

    if idx > 1:
        dependencies = [T[idx//2]]
    else:
        dependencies = []

    @spawn(T[idx], placement=[placement], inout=[active_array], dependencies=dependencies)
    def quicksort_task():

        context = get_current_context()
        print(T[idx], "is running on", context)
        # print("----------------------")
        # print(idx, "Starting Partition on Slice: ", active_slice)
        n_partitions = active_array.nparts
        print("CrossPy has n_partitions: ", n_partitions)

        for arr in active_array.block_view():
            print(arr)
            print(arr.print_overview())



        if n_partitions == 1:
            # print("Base case reached, returning...")
            active_array.values()[0][0].array.sort()
            print(idx, "active_array: ", active_array)

            #NOTE: Global writeback breaks multi-device task guarantees...

            # Can't writeback
            # global_array[active_slice] = active_array
            # xp.alltoallv(active_array, np.arange(len(active_array)), global_array[active_slice])

            return

        pivot_idx = np.random.randint(0, len(active_array))

        # print("Active partition has shape: ", active_array.shape)
        # print("Active partition has len: ", len(active_array))

        # print("The chosen pivot index is: ", pivot_idx)

        pivot = (int)(active_array[pivot_idx])

        # print("The chosen pivot is: ", pivot)

        # local partition
        # print("Performing local partition...")
        splits = partition(active_array, pivot)
        # print("Found the following splits: ", splits)

        local_split = np.sum(splits)
        local_left = local_split
        local_right = len(active_array) - local_split

        # Create new crosspy arrays for the left and right partitions
        # print("Number of elements in the left partition: ", local_left)
        num_left_blocks = math.ceil(local_left / args.m)
        # print("Number of left blocks: ", num_left_blocks)

        # Allocate left array
        left_cupy_blocks = []
        for i in range(num_left_blocks):
            local_length = (int)(min(args.m, local_left - i*args.m))

            with context.devices[i]:
                left_cupy_blocks.append(
                    cp.zeros(local_length, dtype=cp.int32))

        if num_left_blocks == 0:
            left_array = None
        elif num_left_blocks == 1:
            left_array = xp.array(left_cupy_blocks[0], wrapper=parla_wrapper)
        else:
            left_array = xp.array(left_cupy_blocks, dim=0, wrapper=parla_wrapper)

        # print("Number of elements in the right partition: ", local_right)
        num_right_blocks = (int)(math.ceil(local_right / args.m))
        # print("Number of right blocks: ", num_right_blocks)

        # Allocate right array
        right_cupy_blocks = []
        for i in range(num_right_blocks):
            local_length = (int)(min(args.m, local_right - i*args.m))

            with context.devices[i]:
                right_cupy_blocks.append(
                    cp.zeros(local_length, dtype=cp.int32))

        if num_right_blocks == 0:
            right_array = None
        elif num_right_blocks == 1:
            right_array = xp.array(right_cupy_blocks[0], wrapper=parla_wrapper)
        else:
            right_array = xp.array(right_cupy_blocks, dim=0, wrapper=parla_wrapper)

        # print("Left array: ", left_array)
        # print("Left array 0: ", left_array[0:2])
        # print("Right array: ", right_array)
        # print("Right array 0: ", right_array[0:2])

        # Scatter to other partitions
        # print("Performing local scatter...")

        # print("Active array: ", active_array)

        scatter(splits, active_array, left_array, right_array)

        # # form slices to pass to children
        previous_start = active_slice.start
        previous_end = active_slice.stop

        left_start = (int)(previous_start)
        left_end = (int)(previous_start + local_split)
        left_slice = slice(left_start, left_end)

        right_start = (int)(previous_start + local_split)
        right_end = (int)(previous_end)
        right_slice = slice(right_start, right_end)

        if left_array is not None:
            quicksort(2*idx, global_array, left_array, left_slice, T)

        if right_array is not None:
            quicksort(2*idx+1, global_array, right_array, right_slice, T)

def main():

    # Per device size

    global_size = args.m * args.ngpus
    global_array = np.arange(global_size, dtype=np.int32)
    np.random.shuffle(global_array)

    with Parla():
        # Initilize a CrossPy Array
        cupy_list = []

        for i in range(args.ngpus):
            with cp.cuda.Device(i) as device:
                random_array = cp.random.randint(0, 100, size=args.m)
                random_array = random_array.astype(cp.int32)
                cupy_list.append(random_array)
                device.synchronize()

        xA = xp.array(cupy_list, axis=0, wrapper=parla_wrapper)

        print("Original Array: ", xA)
        T = TaskSpace("T")
        start_t = time.perf_counter()
        quicksort(1, xA, xA, slice(0, len(xA)), T)
        end_t = time.perf_counter()

        print("Sorted: ", xA)
        print("Elapsed: ", end_t - start_t)


if __name__ == "__main__":
    main()
