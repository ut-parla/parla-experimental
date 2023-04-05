# from parla import Parla, spawn, TaskSpace
import argparse
import cupy as cp
import numpy as np
import crosspy as xp
import math

parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
parser.add_argument('-num_partitions', type=int, default=2)
parser.add_argument("-num_gpus", type=int, default=1)
parser.add_argument("-m", type=int, default=5)
args = parser.parse_args()

np.random.seed(10)
cp.random.seed(10)

# TODO(wlr): Fuse this kernel
ngpus = args.num_gpus


def partition_kernel(A, B, comp, pivot):
    comp[:] = (A < pivot)
    mid = comp.sum()
    B[:mid] = A[comp]
    B[mid:] = A[~comp]
    A[:] = B[:]
    return mid


def partition(xA, pivot):
    if isinstance(xA, cp.ndarray):
        n_partitions = 1
    else:
        n_partitions = len(xA.values())

    mid = np.zeros(n_partitions+1, dtype=np.uint32)

    for i in range(n_partitions):
        with cp.cuda.Device(i % ngpus):
            if isinstance(xA, cp.ndarray):
                local_array = xA
            else:
                local_array = xA.values()[i]
            workspace = cp.empty_like(local_array)
            comp = cp.empty_like(local_array, dtype=cp.bool_)
            mid[i+1] = partition_kernel(local_array, workspace, comp, pivot)
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
    right_prefix = size_prefix - left_prefix

    left_count = len(left_array)
    right_count = len(right_array)

    print("Size Prefix: ", size_prefix)
    print("Left Prefix: ", left_prefix)
    print("Right Prefix: ", right_prefix)

    left_send_indices = np.zeros(left_count, dtype=np.uint32)
    right_send_indices = np.zeros(right_count, dtype=np.uint32)

    for i in range(n_partitions):
        left_send_indices[left_prefix[i]:left_prefix[i+1]
                          ] = np.arange(size_prefix[i], size_prefix[i]+splits[i+1])
        right_send_indices[right_prefix[i]:right_prefix[i+1]
                           ] = np.arange(size_prefix[i]+splits[i+1], size_prefix[i+1])

    # Performing left scatter
    print("Left send indices: ", left_send_indices)
    print("Right send indices: ", right_send_indices)
    xp.all2allv(active_array, left_send_indices, left_array)

    # Performing right scatter
    right_array[:] = active_array[right_send_indices]
    xp.all2allv(active_array, right_send_indices, right_array)

    return None


def quicksort(global_array, active_array, active_slice, T):

    print("----------------------")
    print("Starting Partition on Slice: ", active_slice)
    if isinstance(active_array, cp.ndarray):
        n_partitions = 1
    else:
        n_partitions = len(active_array.values())
    print("CrossPy has n_partitions: ", n_partitions)

    if len(active_array) == 1:
        print("Base case reached, returning...")
        return

    pivot_idx = np.random.randint(0, len(active_array))

    print("Active partition has shape: ", active_array.shape)
    print("Active partition has len: ", len(active_array))

    print("The chosen pivot index is: ", pivot_idx)

    pivot = (int)(active_array[pivot_idx].to(-1))

    print("The chosen pivot is: ", pivot)

    # local partition
    print("Performing local partition...")
    splits = partition(active_array, pivot)
    print("Found the following splits: ", splits)

    local_split = np.sum(splits)
    local_left = local_split
    local_right = len(active_array) - local_split

    # Create new crosspy arrays for the left and right partitions
    print("Number of elements in the left partition: ", local_left)
    num_left_blocks = math.ceil(local_left / args.m)
    print("Number of left blocks: ", num_left_blocks)

    # Allocate left array
    left_cupy_blocks = []
    for i in range(num_left_blocks):
        local_length = (int)(min(args.m, local_left - i*args.m))

        with cp.cuda.Device(0):
            left_cupy_blocks.append(cp.zeros(local_length, dtype=cp.int32))

    if num_left_blocks == 0:
        left_array = None
    elif num_left_blocks == 1:
        left_array = left_cupy_blocks[0]
    else:
        left_array = xp.array(left_cupy_blocks, dim=0)

    print("Number of elements in the right partition: ", local_right)
    num_right_blocks = (int)(math.ceil(local_right / args.m))
    print("Number of right blocks: ", num_right_blocks)

    # Allocate right array
    right_cupy_blocks = []
    for i in range(num_right_blocks):
        local_length = (int)(min(args.m, local_right - i*args.m))

        with cp.cuda.Device(0):
            right_cupy_blocks.append(cp.zeros(local_length, dtype=cp.int32))

    if num_right_blocks == 0:
        right_array = None
    elif num_right_blocks == 1:
        right_array = right_cupy_blocks[0]
    else:
        right_array = xp.array(right_cupy_blocks, dim=0)

    print("Left array: ", left_array)
    print("Left array 0: ", left_array[0:2])
    print("Right array: ", right_array)
    print("Right array 0: ", right_array[0:2])

    # Scatter to other partitions
    print("Performing local scatter...")

    print("Active array: ", active_array)

    scatter(splits, active_array, left_array, right_array)

    # # form slices to pass to children
    # previous_start = active_slice.start
    # previous_end = active_slice.stop

    # left_start = (int)(previous_start)
    # left_end = (int)(previous_start + local_split)
    # left_slice = slice(left_start, left_end)

    # right_start = (int)(previous_start + local_split)
    # right_end = (int)(previous_end)
    # right_slice = slice(right_start, right_end)

    # quicksort(global_array, left_array, left_slice, T)
    # quicksort(global_array, right_array, right_slice, T)


def main(T):

    # Per device size

    global_size = args.m * args.num_partitions
    global_array = np.arange(global_size, dtype=np.int32)
    np.random.shuffle(global_array)

    # Initilize a CrossPy Array
    cupy_list = []

    for i in range(args.num_partitions):
        with cp.cuda.Device(0):
            random_array = cp.random.randint(0, 100, size=args.m)
            random_array = random_array.astype(cp.int32)
            cupy_list.append(random_array)

    xA = xp.array(cupy_list, dim=0)

    print("Original Array: ", xA)
    quicksort(xA, xA, slice(0, len(xA)), T)

    print("Sorted: ", xA)


if __name__ == "__main__":
    T = None
    main(T)
