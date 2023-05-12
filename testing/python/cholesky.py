"""
A naive implementation of blocked Cholesky using Numba kernels on CPUs.
"""
from parla import Parla, TaskSpace, spawn
import time
from scipy import linalg
import numpy as np
import os
import random

t = 1
os.environ["NUMEXPR_NUM_THREADS"] = str(t)
os.environ["OMP_NUM_THREADS"] = str(t)
os.environ["OPENBLAS_NUM_THREADS"] = str(t)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(t)

# This triangular solve only supports one side
# import cupyx.scipy.linalg as cpx

def numpy_trsm_wrapper(a, b):
    a = np.array(a, order='F', dtype=np.float64)
    b = np.array(b.T, order='F', dtype=np.float64)
    b = linalg.blas.dtrsm(1.0, a, b, trans_a=0, lower=1, side=0)
    return b


def cholesky_inplace(a):
    a = linalg.cholesky(a, lower=True)
    return a


def ltriang_solve(a, b):
    b = numpy_trsm_wrapper(a, b)
    return b.T


def update_kernel(a, b, c):
    c -= a @ b.T
    return c


def update(a, b, c):
    c = update_kernel(a, b, c)
    # c = linalg.blas.dgemm(-1.0, a, b, c=c, beta=1.0, overwrite_c=True, trans_a=False, trans_b=True)
    return c


def clone_here(a):
    return a


def copy(o, i):
    o[:] = i
    return o


def cholesky_blocked_inplace(a, load=1):
    """
    This is a less naive version of dpotrf with one level of blocking.
    Blocks are currently assumed to evenly divide the axes lengths.
    The input array 4 dimensional. The first and second index select
    the block (row first, then column). The third and fourth index
    select the entry within the given block.
    """

    # Define task spaces
    syrk = TaskSpace("syrk")        # Inter-block GEMM
    subcholesky = TaskSpace("subcholesky")  # Cholesky on block
    gemm = TaskSpace("gemm")        # Inter-block GEMM
    solve = TaskSpace("solve")        # Triangular solve
    TaskSpace("zerofy")

    for j in range(len(a)):
        for k in range(j):
            # Inter-block GEMM
            @spawn(syrk[j, k], [solve[j, k], syrk[j, 0:k]], vcus=load)
            # @spawn(syrk[j, k], [solve[j, k]])
            def t0():
                out = clone_here(a[j][j])  # Move data to the current device
                rhs = clone_here(a[j][k])

                out = update(rhs, rhs, out)

                copy(a[j][j], out)  # Move the result to the global array

        # Cholesky on block

        @spawn(subcholesky[j], [syrk[j, 0:j]], vcus=load)
        def t2():
            dblock = clone_here(a[j][j])
            dblock = cholesky_inplace(dblock)
            copy(a[j][j], dblock)

        for i in range(j+1, len(a)):
            for k in range(j):
                # Inter-block GEMM
                @spawn(gemm[i, j, k], [solve[j, k], solve[i, k], gemm[i, j, 0:k]], vcus=load)
                def t3():
                    # Move data to the current device
                    out = clone_here(a[i][j])
                    rhs1 = clone_here(a[i][k])
                    rhs2 = clone_here(a[j][k])

                    out = update(rhs1, rhs2, out)

                    copy(a[i][j], out)  # Move the result to the global array

            # Triangular solve
            @spawn(solve[i, j], [gemm[i, j, 0:j], subcholesky[j]], vcus=load)
            def t4():
                factor = clone_here(a[j][j])
                panel = clone_here(a[i][j])
                panel = ltriang_solve(factor, panel)
                copy(a[i][j], panel)
    return subcholesky[len(a)-1]


def run(matrix='chol_1000.npy', block_size=250, n=1000, check_error=True, check_nan=True, workers=4):

    np.random.seed(10)
    random.seed(10)
    save_file = True
    num_tests = 1
    load = 1.0/workers

    @spawn(vcus=0)
    async def test_blocked_cholesky():
        nonlocal n
        try:
            print("Loading matrix from file: ", matrix)
            a = np.load(matrix)
            print("Loaded matrix from file. Shape=", a.shape)
            n = a.shape[0]
        except Exception:
            print("Generating matrix of size: ", n)
            # Construct input data
            a = np.random.rand(n, n)
            a = a @ a.T

            if save_file:
                np.save(matrix, a)

        # Copy and layout input
        print("Blocksize: ", block_size)
        assert not n % block_size
        a1 = a.copy()
        # a_temp = a1.reshape(n//block_size, block_size, n//block_size, block_size).swapaxes(1, 2)

        for k in range(num_tests):
            ap = a1.copy()
            ap_list = list()
            for i in range(n//block_size):
                ap_list.append(list())
                for j in range(n//block_size):
                    ap_list[i].append(np.copy(
                        a1[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size], order='F'))

            print("Starting Cholesky...")

            start = time.perf_counter()
            # Call Parla Cholesky result and wait for completion
            await cholesky_blocked_inplace(ap_list, load=load)
            end = time.perf_counter()

            print("Completed Cholesky.")

            ts = TaskSpace("CopyBack")

            @ spawn(ts[0])
            def copy_back():
                for i in range(n//block_size):
                    for j in range(n//block_size):
                        ap[i*block_size:(i+1)*block_size, j *
                           block_size: (j+1)*block_size] = ap_list[i][j]

            await ts

            zerofy_start = time.perf_counter()
            computed_L = np.tril(ap)
            zerofy_end = time.perf_counter()

            print("Time:", (end - start) + (zerofy_end - zerofy_start))

            # Check result
            if check_nan:
                is_nan = np.isnan(np.sum(ap))
                print("Is NAN: ", is_nan)
                assert not is_nan

            if check_error:
                error = np.max(np.absolute(a - computed_L @ computed_L.T))
                print("Error", error)
                assert error < 1e-8


if __name__ == '__main__':

    with Parla():
        run()
