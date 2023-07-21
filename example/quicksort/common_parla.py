from typing import Optional

import numpy as np
import cupy as cp

np.random.seed(10)
cp.random.seed(10)

from parla.common.globals import get_current_context
from parla.common.array import copy
from common import partition_kernel


def partition(
    arrays: list,
    pivot,
    *,
    workspace: Optional[list] = None,
    prepend_zero=False,
    unwrap=None
):
    """Partition A against pivot and output to B, return mid index."""
    context = get_current_context()
    n_partitions = len(arrays)
    _z = 1 if prepend_zero else 0  # int(prepend_zero)
    mid = np.zeros(_z + n_partitions, dtype=np.uint32)

    for i, array_in in enumerate(arrays):
        with context.devices[i]:
            array_out = workspace[i] if workspace else cp.empty_like(array_in)
            comp = cp.empty_like(array_in, dtype=cp.bool_)
            mid[_z + i] = partition_kernel(
                unwrap(array_in) if unwrap else array_in,
                unwrap(array_out) if unwrap else array_out,
                pivot, comp=comp
            )
            if workspace is None:
                array_in[:] = array_out[:]
    return mid

def parla_copy(source, source_start, source_end, target, target_start, target_end, *, src_device=None):
    copy(
        target[target_start:target_end],
        source[source_start:source_end]
    )