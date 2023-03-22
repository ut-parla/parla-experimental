import pytest

from parla.common import parray
import numpy as np

def test_parray_creation():
    A = parray.asarray([[1, 2], [3, 4]])

    a = A[0]
    assert A[0,1] == 2
    assert A[1,0] == 3
    assert A[1,1] == 4
    assert np.array_equal(A, np.asarray([[1, 2], [3, 4]]))

if __name__=="__main__":
    test_parray_creation()