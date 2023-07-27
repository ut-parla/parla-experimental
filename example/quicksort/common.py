from typing import Optional

import numpy as np
import cupy as cp
from crosspy import gpu

# Note: this experimental allocator breaks memcpy async
# cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)

def create_array(per_gpu_size, num_gpus, dtype=np.int32, unique=False, return_workspace=True):
    if unique:
        global_array = np.arange(per_gpu_size * num_gpus, dtype=dtype)
        np.random.shuffle(global_array)
    else:
        global_array = np.random.randint(0, 1000000, per_gpu_size * num_gpus).astype(dtype, copy=False)

    # Init a distributed crosspy array
    cupy_list_A = []
    if return_workspace:
        cupy_list_B = []
    for i in range(num_gpus):
        # with cp.cuda.Device(i) as dev:
        with gpu(i) as ctx:
            random_array = cp.asarray(global_array[per_gpu_size * i:per_gpu_size * (i + 1)])
            cupy_list_A.append(random_array)
            if return_workspace:
                cupy_list_B.append(cp.empty(per_gpu_size, dtype=dtype))

    # for i in range(num_gpus):
    #     with cp.cuda.Device(i) as dev:
    #         dev.synchronize()

    if return_workspace:
        return global_array, cupy_list_A, cupy_list_B
    return global_array, cupy_list_A

def get_size_info(array_list):
    """Return size of each array and accumulated sizes
    [0, len(a0), len(a1), ...]
    [0, len(a0), len(a0) + len(a1), ...]
    """
    sizes = np.zeros(len(array_list) + 1, dtype=np.uint32)
    for i, array in enumerate(array_list):
        sizes[i + 1] = len(array)
    return sizes, np.cumsum(sizes)

def partition_kernel(A, B, pivot, *, comp=None):
    if comp is None:
        comp = cp.empty_like(A, dtype=cp.bool_)
    comp[:] = (A < pivot)
    mid = int(comp.sum())
    B[:mid] = A[comp]
    B[mid:] = A[~comp]
    return mid


def partition(
    arrays: list,
    pivot,
    *,
    workspace: Optional[list] = None,
    prepend_zero=False
):
    """Partition A against pivot and output to B, return mid index."""
    n_partitions = len(arrays)
    _z = 1 if prepend_zero else 0  # int(prepend_zero)
    mid = np.zeros(_z + n_partitions, dtype=np.uint32)

    for i, array_in in enumerate(arrays):
        with array_in.device:
            array_out = workspace[i] if workspace else cp.empty_like(array_in)
            comp = cp.empty_like(array_in, dtype=cp.bool_)
            mid[_z + i] = partition_kernel(array_in, array_out, pivot, comp=comp)
            if not workspace: array_in[:] = array_out[:]
    return mid

def scatter(dest, src, left_info, right_info):

    source_starts, target_starts, sizes = left_info

    for source_idx in range(len(dest)):
        for target_idx in range(len(dest)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with dest[target_idx].device:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                # print(source_idx, target_idx, (source_start,
                #      source_end), (target_start, target_end))

                # A[target_idx].array[target_start:target_end] = B[source_idx].array[source_start:source_end]
                target = dest[target_idx]
                source = src[source_idx]
                # print("TARGET: ", target, type(target))
                # print("SOURCE: ", source, type(source))
                target[target_start:target_end] = source[source_start:source_end]

    source_starts, target_starts, sizes = right_info
    for source_idx in range(len(dest)):
        for target_idx in range(len(dest)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with dest[target_idx].device:
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                # print(source_idx, target_idx, (source_start,
                #      source_end), (target_start, target_end))

                # A[target_idx].array[target_start:target_end] = B[source_idx].array[source_start:source_end]
                target = dest[target_idx]
                source = src[source_idx]
                target[target_start:target_end] = source[source_start:source_end]
