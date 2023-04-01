import os
from parla import Parla, spawn, TaskSpace
from parla.cython.device_manager import cpu, gpu
import numpy as np
from parla.common.globals import get_current_context, cupy, DeviceType, CUPY_ENABLED
from parla.common.array import clone_here, copy
from pytest import mark
import pytest

cp = cupy

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


    reference = np.arange(100)
    A = np.zeros_like(reference)
    reference = reference[1:30]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=cpu)
        def main():
            C = np.arange(100)
            C = C[1:30]

            copy(A[1:30], C)

            print(A[1:30])

    assert np.array_equal(reference, A[1:30])


def test_copy_gpu_cpu_into_on_cpu_stride():


    reference = np.arange(100)
    A = np.zeros_like(reference)
    reference = reference[1:30:3]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=cpu)
        def main():
            C = np.arange(100)
            C = C[1:30:3]

            copy(A[1:30:3], C)

            print(A[1:30:3])

    assert np.array_equal(reference, A[1:30:3])


def test_copy_gpu_gpu_into_on_cpu():


    reference = cp.arange(100)
    A = cp.zeros_like(reference)
    reference = reference[1:30]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=cpu)
        def main():
            C = cp.arange(100)
            C = C[1:30]

            copy(A[1:30], C)

            print(A[1:30])

    assert cp.array_equal(reference, A[1:30])


def test_copy_gpu_gpu_into_on_cpu_stride():


    reference = cp.arange(100)
    A = cp.zeros_like(reference)
    reference = reference[1:30:3]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=cpu)
        def main():
            C = cp.arange(100)
            C = C[1:30:3]

            copy(A[1:30:3], C)

            print(A[1:30:3])

    assert cp.array_equal(reference, A[1:30:3])


def test_copy_gpu_gpu_into_on_gpu():


    reference = cp.arange(100)
    A = cp.zeros_like(reference)
    reference = reference[1:30]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=gpu)
        def main():
            C = cp.arange(100)
            C = C[1:30]

            copy(A[1:30], C)

            print(A[1:30])

    assert cp.array_equal(reference, A[1:30])


def test_copy_gpu_gpu_into_on_gpu_stride():


    reference = cp.arange(100)
    A = cp.zeros_like(reference)
    reference = reference[1:30:3]

    B = cp.empty_like(reference)

    with Parla():

        @spawn(placement=gpu)
        def main():
            C = cp.arange(100)
            C = C[1:30:3]

            copy(A[1:30:3], C)

            print(A[1:30:3])

    assert cp.array_equal(A[1:30:3], reference)


def test_clone_here_cpu_gpu_on_gpu():

    A = np.array([[1, 2], [3, 4]])

    with Parla():

        @spawn(placement=gpu)
        def main():
            B = clone_here(A)

            reference = cp.array([[1, 2], [3, 4]])
            assert isinstance(B, cupy.ndarray)
            assert cp.array_equal(reference, B)

def test_clone_here_gpu_gpu_on_gpu():

    A = cp.arange(100)

    with Parla():

        @spawn(placement=gpu)
        def main():
            B = clone_here(A)

            reference = cp.arange(100)
            assert isinstance(B, cupy.ndarray)
            assert cp.array_equal(reference, B)

def test_clone_here_cpu_gpu_on_gpu_stride():

    A = np.arange(100)

    with Parla():

        @spawn(placement=gpu)
        def main():
            B = clone_here(A[1:30:3])

            reference = cp.arange(100)
            reference = reference[1:30:3]

            assert isinstance(B, cupy.ndarray)
            assert cp.array_equal(reference, B)

def test_clone_here_gpu_gpu_on_gpu_stride():

    A = cp.arange(100)

    with Parla():

        @spawn(placement=gpu)
        def main():
            B = clone_here(A[1:30:3])

            reference = cp.arange(100)
            reference = reference[1:30:3]
            assert isinstance(B, cupy.ndarray)
            assert cp.array_equal(reference, B)

