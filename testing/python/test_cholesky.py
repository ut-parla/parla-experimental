import pytest

from parla import Parla

import tempfile
import os

import cholesky


# @pytest.mark.parametrize("block_size", [500, 250, 100])
# def test_cholesky_cpu_1000(block_size):
#     """
#     Run blocked cholesky on cpu
#     """
#     n = 1000
#     with tempfile.TemporaryDirectory() as tempdir:
#         os.path.join(tempdir, f"chol_{n}.npy")
#         filepath = f"chol_{n}.npy"
#         tmplogfile = os.path.join(tempdir, "cholesky.blog")
#         with Parla(logfile=tmplogfile):
#             cholesky.run(matrix=filepath, n=n, block_size=block_size)
#
#
# @pytest.mark.parametrize("block_size", [2000, 1000, 500])
# def test_cholesky_cpu_10000(block_size):
#     """
#     Run blocked cholesky on cpu
#     """
#     n = 10000
#     with tempfile.TemporaryDirectory() as tempdir:
#         os.path.join(tempdir, f"chol_{n}.npy")
#         filepath = f"chol_{n}.npy"
#         tmplogfile = os.path.join(tempdir, "cholesky.blog")
#         with Parla(logfile=tmplogfile):
#             cholesky.run(matrix=filepath, n=n, block_size=block_size)



@pytest.mark.parametrize("block_size", [2000, 1000, 500])
def test_cholesky_cpu_20000(block_size):
    """
    Run blocked cholesky on cpu
    """
    n = 20000
    with tempfile.TemporaryDirectory() as tempdir:
        os.path.join(tempdir, f"chol_{n}.npy")
        filepath = f"chol_{n}.npy"
        tmplogfile = os.path.join(tempdir, "cholesky.blog")
        with Parla(logfile=tmplogfile):
            cholesky.run(matrix=filepath, n=n, block_size=block_size, check_error=False, workers=8)
