# from parla import Parla, spawn, TaskSpace
import argparse
import cupy as cp
import numpy as np
import crosspy as xp
from crosspy import cpu, gpu

parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str, default="devices_sample.YAML")
parser.add_argument("-num_gpus", type=int, default=2)
args = parser.parse_args()

np.random.seed(10)
cp.random.seed(10)

# TODO(wlr): Fuse this kernel


def partition_kernel(A, B, comp, pivot):
    comp[:] = (A < pivot)
    mid = comp.sum()
    B[:mid] = A[comp]
    B[mid:] = A[~comp]
    return mid


def partition(xA, xB, pivot):
    n_partitions = len(xA.values())
    mid = np.zeros(n_partitions+1, dtype=np.uint32)

    for i, (array_in, array_out) in enumerate(zip(xA.values(), xB.values())):
        with cp.cuda.Device(0):
            comp = cp.empty_like(array_in, dtype=cp.bool_)
            mid[i+1] = partition_kernel(array_in, array_out, comp, pivot)
    return mid


# TODO(wlr): Fuse this kernel, pack this better?
def scatter(xA, xB, mid):

    sizes = np.zeros(len(xA.values())+1, dtype=np.uint32)
    for i, array in enumerate(xB.values()):
        sizes[i+1] = len(array)

    size_prefix = np.cumsum(sizes)
    left_prefix = np.cumsum(mid)
    right_prefix = size_prefix - left_prefix
    global_left = np.sum(mid)
    right_prefix += global_left
    print(size_prefix, left_prefix, right_prefix, mid, global_left)

    for i, array in enumerate(xB.values()):
        # Write left
        print("left", left_prefix[i], left_prefix[i+1], mid[i+1])
        if mid[i+1] > 0:
            print("A", xA[left_prefix[i]:left_prefix[i+1]],
                  len(xA[left_prefix[i]:left_prefix[i+1]]))
            print("B local", array[:mid[i+1]], len(array[:mid[i+1]]))
            print("B global", xB[size_prefix[i]:size_prefix[i+1]+mid[i+1]],
                  len(xB[size_prefix[i]:size_prefix[i+1]+mid[i+1]]))

            # QUESTION: How can I perform this copy?
            # xA[left_prefix[i]:left_prefix[i+1]] = array[:mid[i+1]]
            xA[left_prefix[i]:left_prefix[i+1]
               ] = xB[size_prefix[i]:size_prefix[i+1]+mid[i+1]]
        # Write right
        print("right", right_prefix[i], right_prefix[i+1])

        if (sizes[i+1] - mid[i+1]) > 0:
            print("A", xA[right_prefix[i]:right_prefix[i+1]],
                  len(xA[right_prefix[i]:right_prefix[i+1]]))
            print("B local", array[mid[i+1]:], len(array[mid[i+1]:]))
            print("B global", xB[size_prefix[i]+mid[i+1]:size_prefix[i+1]],
                  len(xB[size_prefix[i]+mid[i+1]:size_prefix[i+1]]))

            # QUESTION: How can I perform this copy?
            # xA[right_prefix[i]:right_prefix[i+1]] = array[mid[i+1]:]
            xA[left_prefix[i]:left_prefix[i+1]
               ] = xB[size_prefix[i]+mid[i+1]:size_prefix[i+1]]

    if global_left > 0:
        print("Array left", xA[:global_left])
    if (len(xA) - global_left) > 0:
        print("Array right", xA[global_left:])


def quicksort(xA, xB, slice, T):

    n_partitions = len(xA.values())

    active_A = xA[slice]
    active_B = xB[slice]

    N = len(active_A)
    pivot = (int)(active_A[N-1].to(-1))

    print(N, n_partitions, pivot)

    # local partition
    mid = partition(active_A, active_B, pivot)

    # Scatter to other partitions
    scatter(active_A, active_B, mid)


def main(T):

    # Per device size
    m = 5

    # Initilize a CrossPy Array
    cupy_list_A = []
    cupy_list_B = []
    for i in range(args.num_gpus):
        with cp.cuda.Device(0):
            random_array = cp.random.randint(0, 100, size=m)
            random_array = random_array.astype(cp.int32)

            cupy_list_A.append(random_array)
            cupy_list_B.append(cp.zeros(m, dtype=cp.int32))

    xA = xp.array(cupy_list_A)
    xB = xp.array(cupy_list_B)

    xA = xA.values()[0]
    xB = xB.values()[0]

    quicksort(xA, xB, slice(0, len(xA)), T)


if __name__ == "__main__":
    T = None
    main(T)
