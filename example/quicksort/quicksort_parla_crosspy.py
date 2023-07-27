import nvtx
import numpy as np
import cupy as cp
import crosspy as xp
from crosspy.context import context

import time
from math import log2

from parla import Parla, spawn
from parla.cython.tasks import AtomicTaskSpace
from parla.common.globals import get_current_context
from parla.cython.device_manager import gpu

from common import create_array


def reorder(array, src_gen, T, ptid):
    src = next(src_gen)
    tid = f"{ptid}_{array.device.id}_{len(array)}"  # device id serves as block id
    @spawn(T[tid], placement=[gpu(array.device.id)])
    def _():
        # nvtx.push_range(message=f"{T.name} {ptid} on GPU {array.device.id} {len(array)=}", domain="scatter", color="orange")
        src.tobuffer(array, stream=dict(non_blocking=True))
        # nvtx.pop_range(domain="scatter")

def partition(array, pivot, lout, rout, T, ptid):
    tid = f"{ptid}_{array.device.id}_{len(array)}"
    @spawn(T[tid], placement=[gpu(array.device.id)])
    def _():
        # nvtx.push_range(message=f"{T.name} {ptid} on GPU {array.device.id} {len(array)=}", domain="partition", color="yellow")
        with context(array) as ctx:
            mask = array < pivot
            left = array[mask]
            right = array[~mask]
            if len(left):
                lout[tid] = left
            if len(right):
                rout[tid] = right
        # nvtx.pop_range(domain="partition")


def quicksort(array: xp.CrossPyArray, T, tid=1):
    if args.depth and (tid >> args.depth): return

    if len(array) < 2:
        return

    placement = [gpu(d.id) for d in array.device]
    # print(int(log2(tid)), tid, len(array), len(placement), flush=True)

    @spawn(T[tid], placement=placement)
    async def _():
        # nvtx.push_range(message=f"{T.name} {tid} on {placement} {len(array)=}", domain="quicksort", color="red")
        pivot = int(array[-1])  # without type conversion it's a view, not copy

        lefts, rights = {}, {}
        pT = AtomicTaskSpace(f"partition_{tid}")
        array[:-1].block_view(partition, pivot, lout=lefts, rout=rights, T=pT, ptid=tid)
        await pT
        # chk = [a.device.id for a in rights.values()]
        # assert len(chk) == len(set(chk)), f"{chk=} {list(rights.values())}"
        left = xp.array([a for a in lefts.values() if len(a)], axis=0)
        right = xp.array([a for a in rights.values() if len(a)], axis=0)

        array[len(left)] = pivot
        if len(left):
            aleft = array[:len(left)]
            liter = iter(xp.split(left, aleft.boundaries))
            lT = AtomicTaskSpace(f"left_{tid}")
            aleft.block_view(reorder, liter, T=lT, ptid=tid)
            await lT
            quicksort(aleft, T, 2 * tid)
        if len(right):
            # array[len(left) + 1:] = right[:-1]
            aright = array[len(left) + 1:]
            riter = iter(xp.split(right, aright.boundaries))
            rT = AtomicTaskSpace(f"right_{tid}")
            aright.block_view(reorder, riter, T=rT, ptid=tid)
            await rT
            quicksort(aright, T, 2 * tid + 1)
        # nvtx.pop_range(domain="quicksort")

def main(args, print_heading=True):
    if args.seed:
        np.random.seed(args.seed)
        cp.random.seed(args.seed)

    # Initilize a CrossPy Array
    global_array, cupy_list = create_array(args.m, args.num_gpus, return_workspace=False)
    if not args.verify or args.depth:
        del global_array
    xA = xp.array(cupy_list, axis=0)
    # print("Original Array:\n", xA, flush=True)

    with Parla():
        T = AtomicTaskSpace("quicksort")
        t_start = time.perf_counter()
        quicksort(xA, T)
        T.wait()
        t_end = time.perf_counter()

    # print("Sorted:\n", xA, flush=True)
    if args.verify and not args.depth:
        global_array.sort(kind='quicksort')
        for i in range(len(global_array)):
            assert xA[i] == global_array[i], (i, xA[i], global_array[i])
        print("Result verified")

    stats = dict(
        num_gpus=args.num_gpus,
        per_gpu_size=args.m,
        global_size=args.num_gpus*args.m,
        depth=args.depth,
        time=t_end - t_start
    )
    if print_heading:
        print(','.join(stats.keys()))
    print(','.join(repr(v) for v in stats.values()))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("-m", type=int, default=100, help="Per GPU size")
    parser.add_argument("--verify", default=True, action=argparse.BooleanOptionalAction, help="Verify the sorted result")
    parser.add_argument("-depth", type=int, default=0, help="Max depth of partition recursion (default to 0 as unlimited for full quicksort)")
    parser.add_argument("-seed", type=int, default=0, help="Random seed (default to 0 for no seed)")
    args = parser.parse_args()
    main(args)
