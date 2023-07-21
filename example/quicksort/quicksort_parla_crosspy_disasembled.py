
import numpy as np
import cupy as cp
import crosspy as xp
import math

np.random.seed(10)
cp.random.seed(10)

import time

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace
from parla.common.globals import get_current_context
from parla.cython.device_manager import gpu

from common import create_array, partition_kernel
from common_parla import partition


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
        left_array = active_array[left_send_indices]

    if right_array is not None:
        right_count = len(right_array)
        right_prefix = size_prefix - left_prefix
        right_send_indices = np.zeros(right_count, dtype=np.uint32)
        for i in range(n_partitions):
            right_send_indices[right_prefix[i]:right_prefix[i+1]
                               ] = np.arange(size_prefix[i]+splits[i+1], size_prefix[i+1])

        # Performing right scatter
        right_array = active_array[right_send_indices]

    return None


def quicksort(active_array, active_slice, T, Tid=1):

    n_partitions = active_array.nparts

    if n_partitions == 1:
        dev_id = next(iter(active_array.device_array.values())).device.id
        print("CREATING TASK CONSTRAINTS: ", Tid, active_array, dev_id)
        placement = gpu(dev_id)[{'vcus': 1000}]
    else:
        placement = tuple(
                [gpu(arr.device.id)[{'vcus': 1000}] for arr in active_array.block_view()])

        print("CREATING TASK CONSTRAINTS: ", Tid, active_array, [arr.device.id for arr in active_array.block_view()])

    active_array = list(active_array.device_array.values())
    print("Placement for : ", Tid, placement, flush=True)

    @spawn(T[Tid], placement=[placement])
    def quicksort_task():
        context = get_current_context()
        print(T[Tid], "is running on", context)
        # print("----------------------")
        # print(idx, "Starting Partition on Slice: ", active_slice)
        nonlocal n_partitions
        # print("CrossPy has n_partitions: ", n_partitions)

        if n_partitions == 1:
            # print("Base case reached, returning...")
            next(iter(active_array.device_array.values())).sort()
            print(Tid, "active_array: ", active_array)
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
        splits = partition(active_array, pivot, prepend_zero=True)
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

        left_array = xp.array(left_cupy_blocks, dim=0)

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

        right_array = xp.array(right_cupy_blocks, axis=0)

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
            quicksort(left_array, left_slice, T, 2*Tid)

        if right_array is not None:
            quicksort(right_array, right_slice, T, 2*Tid+1)

def main(args):
    global_array, cupy_list, _ = create_array(args.m, args.num_gpus)

    # Initilize a CrossPy Array
    xA = xp.array(cupy_list, axis=0)

    print("Original Array: ", xA, flush=True)

    with Parla():
        T = AtomicTaskSpace("T")
        t_start = time.perf_counter()
        quicksort(xA, slice(0, len(xA)), T)
        t_end = time.perf_counter()

    print("Sorted:")
    print(xA)
    print("Time: ", t_end - t_start)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
    parser.add_argument("-num_gpus", type=int, default=2)
    parser.add_argument("-m", type=int, default=10)
    args = parser.parse_args()
    main(args)
