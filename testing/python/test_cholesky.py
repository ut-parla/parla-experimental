import pytest

from parla import Parla
from parla.utility.graphs import read_pgraph, parse_blog
from parla.utility.execute import timeout

import tempfile
import os
from ast import literal_eval as make_tuple

import cholesky


@pytest.mark.parametrize("block_size", [500, 250, 100])
def test_cholesky_cpu_1000(block_size):
    """
    Run blocked cholesky on cpu
    """
    n = 1000
    with tempfile.TemporaryDirectory() as tempdir:
        tmpfilepath = os.path.join(tempdir, f"chol_{n}.npy")
        filepath = f"chol_{n}.npy"
        tmplogfile = os.path.join(tempdir, f"cholesky.blog")
        with Parla(logfile=tmplogfile):
            cholesky.run(matrix=filepath, n=n, block_size=block_size)


@pytest.mark.parametrize("block_size", [2000, 1000, 500])
def test_cholesky_cpu_10000(block_size):
    """
    Run blocked cholesky on cpu
    """
    n = 10000
    with tempfile.TemporaryDirectory() as tempdir:
        tmpfilepath = os.path.join(tempdir, f"chol_{n}.npy")
        filepath = f"chol_{n}.npy"
        tmplogfile = os.path.join(tempdir, f"cholesky.blog")
        with Parla(logfile=tmplogfile):
            cholesky.run(matrix=filepath, n=n, block_size=block_size)
