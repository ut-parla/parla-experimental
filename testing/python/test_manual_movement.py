import os
from parla import Parla, spawn, TaskSpace
from parla.cython.device_manager import cpu, cuda
import numpy as np
from parla.common.globals import get_current_context, cupy, DeviceType, CUPY_ENABLED
from parla.common.array import clone_here, copy
from pytest import mark
import pytest

cp = cupy
gpu = cuda

if CUPY_ENABLED:
    num_gpus = cp.cuda.runtime.getDeviceCount()
else:
    num_gpus = 0
    # FIXME: Disable tests that require cupy.


def test_clone_here_cpu_cpu_on_cpu():

    A = np.array([[1, 2], [3, 4]])

    with Parla():

        @spawn(placement=cpu)
        def main():
            B = clone_here(A)

            assert isinstance(B, np.ndarray)
            assert np.array_equal(A, B)


def test_clone_here_gpu_cpu_on_cpu():

    A = cp.array([[1, 2], [3, 4]])

    reference = np.array([[1, 2], [3, 4]])

    with Parla():

        @spawn(placement=cpu)
        def main():
            B = clone_here(A)

            assert isinstance(B, np.ndarray)
            assert np.array_equal(reference, B)


def test_clone_here_gpu_cpu_on_gpu_stride():

    A = cp.arange(100)

    reference = np.arange(100)
    reference = reference[1:30:3]

    with Parla():

        @spawn(placement=cpu)
        def main():
            B = clone_here(A[1:30:3])

            assert isinstance(B, np.ndarray)
            assert np.array_equal(reference, B)


def test_copy_gpu_cpu_into_on_cpu():

    A = cp.zeros(100)

    reference = np.arange(100)
    reference = reference[1:30:3]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=cpu)
        def main():
            C = np.arange(100)
            C = C[1:30]

            copy(A[1:30], C)

            print(A[1:30])


def test_copy_gpu_cpu_into_on_cpu_stride():

    A = cp.zeros(100)

    reference = np.arange(100)
    reference = reference[1:30:3]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=cpu)
        def main():
            C = np.arange(100)
            C = C[1:30:3]

            copy(A[1:30:3], C)

            print(A[1:30:3])


def test_copy_gpu_gpu_into_on_cpu():

    A = cp.zeros(100)

    reference = np.arange(100)
    reference = reference[1:30:3]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=cpu)
        def main():
            C = cp.arange(100)
            C = C[1:30]

            copy(A[1:30], C)

            print(A[1:30])


def test_copy_gpu_gpu_into_on_cpu_stride():

    A = cp.zeros(100)

    reference = np.arange(100)
    reference = reference[1:30:3]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=cpu)
        def main():
            C = cp.arange(100)
            C = C[1:30:3]

            copy(A[1:30:3], C)

            print(A[1:30:3])


def test_copy_gpu_gpu_into_on_gpu():

    A = cp.zeros(100)

    reference = np.arange(100)
    reference = reference[1:30:3]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=gpu)
        def main():
            C = cp.arange(100)
            C = C[1:30]

            copy(A[1:30], C)

            print(A[1:30])


def test_copy_gpu_gpu_into_on_gpu_stride():

    A = cp.zeros(100)

    reference = np.arange(100)
    reference = reference[1:30:3]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=gpu)
        def main():
            C = cp.arange(100)
            C = C[1:30:3]

            copy(A[1:30:3], C)

            print(A[1:30:3])


def test_clone_here_cpu_gpu_on_gpu():

    A = np.array([[1, 2], [3, 4]])
    reference = cp.array([[1, 2], [3, 4]])

    with Parla():

        @spawn(placement=gpu)
        def main():
            B = clone_here(A)

            assert isinstance(B, cupy.ndarray)
            assert cp.array_equal(reference, B)


def test_clone_here_cpu_gpu_on_gpu_stride():

    A = np.arange(100)
    reference = cp.arange(100)
    reference = reference[1:30:3]

    with Parla():

        @spawn(placement=gpu)
        def main():
            B = clone_here(A[1:30:3])

            assert isinstance(B, cupy.ndarray)
            assert cp.array_equal(reference, B)


def test_clone_here_gpu_gpu_on_gpu():

    A = cp.arange(100)
    reference = cp.arange(100)

    with Parla():

        @spawn(placement=gpu)
        def main():
            B = clone_here(A)

            assert isinstance(B, cupy.ndarray)
            assert cp.array_equal(reference, B)


@mark.parametrize("i", range(num_gpus))
def test_clone_here_gpu_gpu_on_gpu_stride(i):

    with cp.cuda.Device(i):
        A = cp.arange(100)
        reference = cp.arange(100)
        reference = reference[1:30:3]

    with Parla():

        @spawn(placement=gpu)
        def main():
            B = clone_here(A[1:30:3])

            assert isinstance(B, cupy.ndarray)
            assert cp.array_equal(reference, B)