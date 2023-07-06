import numpy as np
import cupy as cp

np.random.seed(10)
cp.random.seed(10)

from parla.common.globals import get_current_context
from common import partition_kernel


def partition(A, B, pivot):
    context = get_current_context()
    n_partitions = len(A)
    mid = np.zeros(n_partitions, dtype=np.uint32)

    for i, array_in in enumerate(A):
        with context.devices[i]:
            comp = cp.empty_like(array_in, dtype=cp.bool_)
            array_out = cp.empty_like(array_in)
            print("Locations: ", comp.device, array_in.device, array_out.device)
            mid[i] = partition_kernel(array_in, array_out, comp, pivot)
            B[i] = array_out
    return mid



def scatter(A, B, left_info, right_info):
    context = get_current_context()

    source_starts, target_starts, sizes = left_info

    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with context.devices[target_idx]:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                target = A[target_idx]
                source = B[source_idx]

                with context.devices[source_idx]:
                    temp = cp.ascontiguousarray(source[source_start:source_end])

                target[target_start:target_end] = cp.asarray(temp)

    source_starts, target_starts, sizes = right_info
    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with context.devices[target_idx]:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                target = A[target_idx]
                source = B[source_idx]

                print("TARGET: ", target, type(target))
                print("SOURCE: ", source, type(source))

                with context.devices[source_idx]:
                    temp = cp.ascontiguousarray(source[source_start:source_end])

                target[target_start:target_end] = cp.asarray(temp)
