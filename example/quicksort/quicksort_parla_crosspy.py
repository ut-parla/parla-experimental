
import numpy as np
import cupy as cp
import crosspy as xp

np.random.seed(10)
cp.random.seed(10)

import time

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace
from parla.common.globals import get_current_context
from parla.cython.device_manager import gpu

from common import create_array


def quicksort(array: xp.CrossPyArray, T, Tid=1):
    placement = tuple(array.block_view(lambda arr: gpu(arr.device.id)))
    # print(Tid, array, placement, flush=True)

    @spawn(T[Tid], placement=[placement])
    async def quicksort_task():
        if len(array) < 2:
            return

        pivot = int(array[len(array) - 1])  # without type conversion it's a view, not copy
        left_mask = (array < pivot)
        left = array[left_mask]
        right_mask = ~left_mask
        right_mask[len(array) - 1] = False
        right = array[right_mask]
        assert len(array) == len(left) + 1 + len(right)

        array[len(left)] = pivot
        if len(left):
            array[:len(left)] = left
            await quicksort(array[:len(left)], T, 2 * Tid)
        if len(right):
            array[len(left) + 1:] = right
            await quicksort(array[len(left) + 1:], T, 2 * Tid + 1)

    return T[Tid]

def main(args):
    global_array, cupy_list, _ = create_array(args.m, args.num_gpus)

    # Initilize a CrossPy Array
    xA = xp.array(cupy_list, axis=0)

    print("Original Array: ", xA, flush=True)

    with Parla():
        T = AtomicTaskSpace("T")
        t_start = time.perf_counter()
        quicksort(xA, T)
        T.wait()
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
