from parla import Parla, spawn, TaskSpace
from parla.common.globals import get_current_context
from parla.cython.device_manager import cpu, gpu

import argparse
import cupy as cp
import numpy as np
import crosspy as xp
import time

parser = argparse.ArgumentParser()
parser.add_argument("-ngpus", type=int, default=2)
parser.add_argument("-m", type=int, default=3)
args = parser.parse_args()

np.random.seed(10)
cp.random.seed(10)

ngpus = args.ngpus

def quicksort(array: xp.CrossPyArray, out, T, T_idx=0):
    placement = tuple(array.block_view(lambda arr: gpu(arr.device.id)[{'vcus': 1000}]))
    print("CREATING TASK CONSTRAINTS: ", T_idx, array, array.block_view(lambda arr: arr.device.id))
    print("Placement for : ", T_idx, placement, flush=True)

    @spawn(T[T_idx], placement=[placement])
    def quicksort_task():
        context = get_current_context()
        print(T[T_idx], "is running on", context)

        if len(array) < 2:
            if out is not array:
                out[:] = array
            return

        pivot = int(array[len(array) - 1])  # without type conversion it's a view, not copy
        left_mask = (array < pivot)
        left = array[left_mask]
        right_mask = ~left_mask
        right_mask[len(array) - 1] = False
        right = array[right_mask]
        assert len(array) == len(left) + 1 + len(right)

        if len(left):
            quicksort(left, left, T, 2*T_idx)
            xp.assignment(out, cp.arange(len(left)), left, None)
        if len(right):
            quicksort(right, right, T, 2*T_idx+1)
            xp.assignment(out, cp.arange(len(left) + 1, len(array)), right, None)
        out[len(left)] = pivot

def main():
    with Parla():
        # Initilize a CrossPy Array
        cupy_list = []
        for i in range(args.ngpus):
            with cp.cuda.Device(i) as device:
                random_array = cp.random.randint(0, 100, size=args.m)
                random_array = random_array.astype(cp.int32)
                cupy_list.append(random_array)
                device.synchronize()

        xA = xp.array(cupy_list, axis=0)
        print("Original Array: ", xA)
        T = TaskSpace("T")
        start_t = time.perf_counter()
        quicksort(xA, xA, T)
        end_t = time.perf_counter()

    print("Sorted: ", xA)
    print("Elapsed: ", end_t - start_t)


if __name__ == "__main__":
    main()
