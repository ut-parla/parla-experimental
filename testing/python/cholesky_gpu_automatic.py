"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""

from parla.common.parray import asarray
from cupy.cuda import device
from cupy.cuda import cublas
import cupy as cp
from parla import Parla, TaskSpace, spawn

from parla.cython.device_manager import cpu
from parla.cython.device_manager import gpu
from parla.common.globals import get_current_devices
from parla.common.globals import PyMappingPolicyType

import time
import numpy as np
import random
import argparse
import os

parser = argparse.ArgumentParser()
# Blocksize
parser.add_argument('-b', type=int, default=2000)
# How many blocks
parser.add_argument('-nblocks', type=int, default=14)
# How many trials to run
parser.add_argument('-trials', type=int, default=1)
# What matrix file (.npy) to load
parser.add_argument('-matrix', default=None)
# Are the placements fixed by the user or determined by the scheduler?
parser.add_argument('-fixed', default=0, type=int)
# How many GPUs to run on?
parser.add_argument('-ngpus', default=4, type=int)
args = parser.parse_args()

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = cuda_visible_devices.strip().split(',')
    cuda_visible_devices = list(map(int, cuda_visible_devices))

gpus = cuda_visible_devices[:args.ngpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


ngpus = cp.cuda.runtime.getDeviceCount()
# Make sure that the enviornment configuration is correct
assert (ngpus == args.ngpus)

block_size = args.b
fixed = args.fixed

if args.matrix is None:
    n = block_size*args.nblocks

num_tests = args.trials

loc = gpu
v = 1000
save_file = True
check_nan = True
check_error = False
time_zeros = False  # Set to true if comparing with Dask.

loc = gpu


def cholesky(a):
    a = cp.linalg.cholesky(a)
    # if cp.any(cp.isnan(a)):
    #  raise np.linalg.LinAlgError
    return a


def cupy_trsm_wrapper(a, b):
    cublas_handle = device.get_cublas_handle()
    trsm = cublas.dtrsm
    uplo = cublas.CUBLAS_FILL_MODE_LOWER
    a = cp.asarray(a, dtype=np.float64, order='F')
    b = cp.asarray(b, dtype=np.float64, order='F')
    trans = cublas.CUBLAS_OP_T
    side = cublas.CUBLAS_SIDE_RIGHT

    # trans = cublas.CUBLAS_OP_T
    # side = cublas.CUBLAS_SIDE_LEFT

    diag = cublas.CUBLAS_DIAG_NON_UNIT
    m, n = (b.side, 1) if b.ndim == 1 else b.shape
    alpha = np.array(1, dtype=a.dtype)
    # Cupy >= 9 requires pointers even for coefficients.
    # https://github.com/cupy/cupy/issues/7011
    trsm(cublas_handle, side, uplo, trans, diag, m, n,
         alpha.ctypes.data, a.data.ptr, m, b.data.ptr, m)
    return b


def ltriang_solve(a, b):
    b = cupy_trsm_wrapper(a, b)
    return b


def update_kernel(a, b, c):
    c -= a @ b.T
    return c


def update(a, b, c):
    c = update_kernel(a, b, c)
    return c


def flatten(t):
    return [item for sublist in t for item in sublist]


def cholesky_blocked_inplace(a, block_size, trial):
    """
    This is a less naive version of dpotrf with one level of blocking.
    Blocks are currently assumed to evenly divide the axes lengths.
    The input array 4 dimensional. The first and second index select
    the block (row first, then column). The third and fourth index
    select the entry within the given block.
    """
    # TODO (bozhi): these should be guaranteed by the partitioner
    # if len(a) * a[0][0].shape[0] != len(a[0]) * a[0][0].shape[1]:
    #    raise ValueError("A square matrix is required.")
    # if len(a) != len(a[0]):
    #    raise ValueError("Non-square blocks are not supported.")

    # print("Starting..", flush=True)
    # print("Initial Array", a, flush=True)
    # block_size = a[0][0].array.shape[0]
    # Define task spaces
    gemm1 = TaskSpace("gemm1-" + trial)        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky-", trial)  # Cholesky on block
    gemm2 = TaskSpace("gemm2-" + trial)        # Inter-block GEMM
    solve = TaskSpace("solve-" + trial)        # Triangular solve

    for j in range(len(a)):
        for k in range(j):
            # Inter-block GEMM

            loc_syrk = gpu[{"vcus":v}]
            if fixed:
                print("Fixed is enabled?", flush=True)
                loc_syrk = gpu(j % ngpus)[{'vcus': v}]

            @spawn(gemm1[j, k], [solve[j, k], gemm1[j, 0:k]], input=[(a[j][k], 0)], inout=[(a[j][j], 0)], placement=[loc_syrk])
            def t1():
                # print(f"+SYRK: ({j}, {k}) - Requires rw({j},{j})  r({j}, {k})", get_current_devices(), flush=True)
                out = a[j][j].array
                rhs = a[j][k].array
                out = update(rhs, rhs, out)

                stream = cp.cuda.get_current_stream()
                stream.synchronize()

                # print(f"==SYRK: ({j}, {k}) - Requires rw({j},{j})  r({j}, {k})", out.device.id, rhs.device.id, a[j][j].array.device.id, flush=True)
                # a[j][j].update(out)
                stream.synchronize()
                # print(
                #    f"-SYRK: ({j}, {k}) - Requires rw({j},{j})  r({j}, {k})", flush=True)

        # Cholesky on block

        loc_potrf = gpu[{"vcus":v}]
        if fixed:
            loc_potrf = gpu(j % ngpus)[{'vcus': v}]

        @spawn(subcholesky[j], [gemm1[j, 0:j]], inout=[(a[j][j], 0)], placement=[loc_potrf])
        def t2():
            # print(f"+POTRF: ({j}) - Requires rw({j},{j})", get_current_devices(), flush=True)
            dblock = a[j][j].array

            dblock = cholesky(dblock)

            stream = cp.cuda.get_current_stream()
            stream.synchronize()

            # print(f"==POTRF: ({j}) - Requires rw({j},{j}) Locations: ", dblock.device.id, a[j][j].device.id, cp.cuda.runtime.getDevice(), flush=True)
            # a[j][j].update(dblock)
            stream.synchronize()
            # print(f"-POTRF: ({j}) - Requires rw({j},{j})", flush=True)
        for i in range(j+1, len(a)):
            for k in range(j):
                # Inter-block GEMM

                loc_gemm = gpu[{"vcus":v}]
                if fixed:
                    loc_gemm = gpu(i % ngpus)[{'vcus': v}]

                @spawn(gemm2[i, j, k], [solve[j, k], solve[i, k], gemm2[i, j, 0:k]], inout=[(a[i][j], 0)], input=[(a[i][k], 0), (a[j][k], 0)], placement=[loc_gemm])
                def t3():
                    # print(f"+GEMM: ({i}, {j}, {k}) - Requires rw({i},{j}), r({i}, {k}), r({j}, {k})", get_current_devices(), flush=True)
                    out = a[i][j].array
                    rhs1 = a[i][k].array
                    rhs2 = a[j][k].array

                    stream = cp.cuda.get_current_stream()

                    out = update(rhs1, rhs2, out)
                    stream.synchronize()

                    # print(f"==GEMM: ({i}, {j}, {k}) - Requires rw({i},{j}), r({i}, {k}), r({j}, {k}) Locations", out.device.id, rhs1.device.id, rhs2.device.id, a[i][j].array.device.id, cp.cuda.runtime.getDevice(), flush=True)
                    # a[i][j].update(out)
                    stream.synchronize()
                    # print(f"-GEMM: ({i}, {j}, {k}) - Requires rw({i},{j}), r({i}, {k}), r({j}, {k})", get_current_devices(), flush=True)

            # Triangular solve

            loc_trsm = gpu[{"vcus":v}]
            if fixed:
                loc_trsm = gpu(i % ngpus)[{'vcus': v}]

            @spawn(solve[i, j], [gemm2[i, j, 0:j], subcholesky[j]], inout=[(a[i][j], 0)], input=[(a[j][j], 0)], placement=[loc_trsm])
            def t4():
                # print(f"+TRSM: ({i}, {j}) - Requires rw({i},{j}), r({j}, {j})", get_current_devices(), cp.cuda.runtime.getDevice(), flush=True)
                factor = a[j][j].array
                panel = a[i][j].array

                out = ltriang_solve(factor, panel)
                stream = cp.cuda.get_current_stream()
                stream.synchronize()
                # print(f"==TRSM: ({i}, {j}) - Requires rw({i},{j}), r({j}, {j}) Locations", factor.device.id, panel.device.id, out.device.id, a[i][j].device.id, cp.cuda.runtime.getDevice(), flush=True)
                # a[i][j].update(out)
                a[i][j].array[:] = out
                stream.synchronize()
                # print(
                #    f"-TRSM: ({i}, {j}) - Requires rw({i},{j}), r({j}, {j})", flush=True)

    return subcholesky[len(a) - 1]


def main():
    @spawn(placement=cpu)
    async def test_blocked_cholesky():
        global n

        if args.matrix is None:
            print("Generating matrix of size: ", n)
            np.random.seed(10)
            # Construct input data
            a = np.random.rand(n, n)
            a = a @ a.T

            if save_file:
                np.save(f"chol_{n}", a)
        else:
            print("Loading matrix from file: ", args.matrix)
            a = np.load(args.matrix)
            print("Loaded matrix from file. Shape=", a.shape)
            n = a.shape[0]

        # Copy and layout input
        print("Blocksize: ", block_size)
        assert not n % block_size
        a1 = a.copy()
        # a_temp = a1.reshape(n//block_size, block_size, n//block_size, block_size).swapaxes(1, 2)

        n_gpus = cp.cuda.runtime.getDeviceCount()
        ap_parray = None
        ap_list = None

        for k in range(num_tests):
            ap = a1.copy()

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

            begin_rl_ts = TaskSpace("begin_rl_task")
            end_rl_ts = TaskSpace("end_rl_task")

            @spawn(begin_rl_ts[0])
            def begin_rl_task():
                pass

            await begin_rl_ts[0]


            if k == 0:
                ap_list = list()
                for i in range(n//block_size):
                    ap_list.append(list())
                    for j in range(n//block_size):
                        with cp.cuda.Device(i % n_gpus):
                            ap_list[i].append(cp.asarray(
                                a1[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size], order='F'))
                            cp.cuda.Device().synchronize()

                ap_parray = []
                for i in range(n//block_size):
                    ap_parray.append([])
                    for j in range(n//block_size):
                        ap_parray[i].append(
                            asarray(ap_list[i][j], name=f"ap_{i}_{j}"))

            else:
                rs = TaskSpace("Reset-" + str(k))
                for i in range(n//block_size):
                    for j in range(n//block_size):
                        @spawn(rs[i, j], placement=gpu(i % n_gpus), inout=[(ap_parray[i][j], 0)])
                        def reset():
                            ap_parray[i][j].array[:] = cp.asarray(
                                a1[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size], order='F')

                            print("i, j size:", ap_parray[i][j].nbytes)
                            cp.cuda.stream.get_current_stream().synchronize()

                await rs

            print("Starting Cholesky", flush=True)
            print("------------", flush=True)
            start = time.perf_counter()

            # Call Parla Cholesky result and wait for completion
            await cholesky_blocked_inplace(ap_parray, block_size, str(k))
            # print(ap_parray)

            # print(ap)
            end = time.perf_counter()

            ts = TaskSpace("CopyBack-" + str(k))
            plist = flatten(ap_parray)
            plist = [(p, 0) for p in plist]

            @spawn(ts[0], placement=cpu, input=plist)
            def copy_back():
                for i in range(n//block_size):
                    for j in range(n//block_size):
                        ap[i*block_size:(i+1)*block_size, j*block_size:(j+1)
                           * block_size] = ap_parray[i][j].array
            await ts

            if time_zeros:
                zerofy_start = time.perf_counter()
                computed_L_cupy = cp.tril(cp.array(ap))
                zerofy_end = time.perf_counter()
            else:
                zerofy_start = 0
                zerofy_end = 0

            print("Time:", (end - start) + (zerofy_end - zerofy_start))

            print("--------")
            # Check result
            print("Is NAN: ", np.isnan(np.sum(ap)))
            if np.isnan(np.sum(ap)) is True:
                print(ap)

            if check_error:
                if time_zeros:
                    computed_L = cp.asnumpy(computed_L_cupy)
                else:
                    computed_L = np.tril(ap)
                print(computed_L)
                error = np.max(np.absolute(a - computed_L @ computed_L.T))
                print("Error", error)


            @spawn(end_rl_ts[0])
            def end_rl_task():
                pass

            await end_rl_ts[0]

if __name__ == '__main__':
    np.random.seed(10)
    random.seed(10)
    with Parla(mapping_policy = PyMappingPolicyType.RLTest):
#with Parla():
        main()
