import argparse 
parser = argparse.ArgumentParser()
#Size of matrix
parser.add_argument('-n', type=int, default=4, help='Size of matrix')
parser.add_argument('-k', type=int, default=4, help='Size of matrix')
parser.add_argument('-m', type=int, default=4, help='Size of matrix')
#How many gpus to use
parser.add_argument('-ngpus', type=int, default=2)
args = parser.parse_args()

from parla import Parla
from parla.devices import cpu, gpu
from parla.tasks import spawn, TaskSpace
from parla.array import clone_here

from parla.tasks import get_current_context
import crosspy as xp
import numpy as np
import cupy as cp
from cufftmg import handler

import time

sizes = [1000, 1000, 1000]
max_alloc = 1000

def construct_local_xpy(xA, device_ids):

    local_A = []
    for i, block in enumerate(xA.block_view()):
        if i in device_ids:
            local_A.append(block)

    tA = xp.array(local_A, axis=0)
    return tA


def launch_fft(T, idx, xA, placement, n, diff=True):
    @spawn(T[idx], placement=placement, vcus=1)
    def task():
        context = get_current_context()
        print(f"+Task {idx} on {context}", flush=True)
        device_ids = context.gpu_ids
        streams = context.streams

        if diff:
            tA = clone_here(xA)
        else:
            tA = construct_local_xpy(xA, device_ids)

        
        for stream in streams:
            stream.synchronize()

        print(f"=Task {idx}", flush=True)

        h = handler()
        h.configure(device_ids=device_ids, streams=streams, size=n)
        h.fft2(tA)
        h.ifft2(tA)

        for stream in streams:
            stream.synchronize()

        print(array)

        print(f"-Task {idx}", flush=True)


def create_matricies(n, device_set, part=True):
    if part:
        d = n // len(device_set)
    else:
        d = n

    A_list = []

    for i in device_set:
        #For the moment we assume that the matrices are square
        if i < 0:
            A_list.append(np.ones((n,d), dtype=np.float64))
        else:
            with cp.cuda.Device(i) as dev:
                A_list.append(cp.ones((n,d), dtype=np.float64))
                dev.synchronize()

    xA = xp.array(A_list, axis=0)

    return xA

def ngpus_to_size(n):
    if n == 1:
        return sizes[0]
    elif n == 2:
        return sizes[1]
    elif n==4:
        return sizes[2]

def initialize_matrix_list(submit_list, diff=True):
    matrix_list = []
    if diff:
        for ndevices in submit_list:
            n = ngpus_to_size(ndevices)
            device_set = [-1 for i in range(ndevices)]
            xA = create_matricies(n, device_set)
            matrix_list.append(Ax)
    else:
        max_size = max_alloc
        device_set = [i for i in range(4)]
        xA = create_matricies(max_size, device_set)
        for ndevices in submit_list:
            matrix_list.append(xA)

    return matrix_list


def main():

    @spawn(placement=cpu, vcus=0)
    async def main():

        T = TaskSpace("T")

        #submit_list = [1, 4, 2, 1, 1, 2, 1]
        submit_list = [2]

        diff=False

        #For now have everything uses the same matrix
        matrix_list = initialize_matrix_list(submit_list, diff=diff)

        start_t = time.perf_counter()

        for i, matrix_set in enumerate(matrix_list):
            xA = matrix_set
            ndevices = submit_list[i]
            placement = [gpu*ndevices]
            n = ngpus_to_size(ndevices)
            launch_fft(T, i, xA, placement, n, diff=diff)

        await T

        end_t = time.perf_counter()
        elapsed = end_t - start_t 

        print(f"Total Time: {elapsed}", flush=True)


if __name__ == "__main__":
    with Parla():
        main()
