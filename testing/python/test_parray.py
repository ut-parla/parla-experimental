import pytest

from parla import Parla, spawn, TaskSpace, parray
from parla.common.parray.coherence import Coherence
from parla.cython.device_manager import cpu, cuda
import numpy as np

def test_parray_creation():
    A = parray.asarray([[1, 2], [3, 4]])

    a = A[0]
    assert A[0,1] == 2
    assert A[1,0] == 3
    assert A[1,1] == 4
    assert np.array_equal(A, np.asarray([[1, 2], [3, 4]]))

def test_parray_task():
    with Parla():
        @spawn(placement=cpu)
        def main():
            n = 2
            np.random.seed(10)
            # Construct input data
            a = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            b = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            a = parray.asarray(a)
            b = parray.asarray(a)

            ts = TaskSpace("CopyBack")

            @spawn(ts[1], placement=cuda(1))
            def check_array_write():
                b._auto_move(1, do_write = True)
                assert b[0,0] == 1
                assert b._current_device_index == 1
                
                b[1,1] = 0
                assert b[1,1] == 0
                assert b._array._buffer[1] is not None
                assert b._array._buffer[-1] is None
                assert b._coherence._local_states[-1] == Coherence.INVALID
                assert b._coherence._local_states[1] == Coherence.MODIFIED


                assert a._current_device_index == 1
                assert a._array._buffer[1] is None
                assert a._array._buffer[-1] is not None
                assert a._coherence._local_states[1] == Coherence.INVALID
                assert a._coherence._local_states[-1] == Coherence.MODIFIED

            @spawn(ts[2], dependencies=[ts[1]], placement=cuda(0))
            def check_array_slicing():
                a[1]._auto_move(0, do_write = True)
                assert a[1,0] == 1
                assert a._current_device_index == 0
                
                a[1,1] = 0
                assert a[1,1] == 0
                assert a._array._buffer[-1] is not None
                assert isinstance(a._array._buffer[0], list)
                assert a._coherence._local_states[-1] == Coherence.INVALID
                assert isinstance(a._coherence._local_states[0], dict)

            @spawn(ts[3], dependencies=[ts[2]], placement=cpu)
            def check_array_write_back():
                a._auto_move(-1, do_write = True)
                assert a[1,1] == 0
                assert a._current_device_index == -1
                
                assert a._array._buffer[-1] is not None
                assert a._array._buffer[0] is None
                assert a._coherence._local_states[-1] == Coherence.MODIFIED
                assert a._coherence._local_states[0] == Coherence.INVALID

if __name__=="__main__":
    test_parray_creation()
    test_parray_task()