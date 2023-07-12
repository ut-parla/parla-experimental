from typing import Optional

import numpy as np
import cupy as cp

np.random.seed(10)
cp.random.seed(10)

# Note: this experimental allocator breaks memcpy async
# cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)

def create_array(per_gpu_size, num_gpus, unique=False):
    if unique:
        global_array = np.arange(per_gpu_size * num_gpus, dtype=np.int32)
        np.random.shuffle(global_array)
    else:
        global_array = cp.random.randint(0, 1000000, per_gpu_size * num_gpus).astype(cp.int32)

    # Init a distributed crosspy array
    cupy_list_A = []
    cupy_list_B = []
    for i in range(num_gpus):
        with cp.cuda.Device(i) as dev:
            random_array = cp.asarray(global_array[per_gpu_size * i:per_gpu_size * (i + 1)])
            cupy_list_A.append(random_array)
            cupy_list_B.append(cp.empty(per_gpu_size, dtype=cp.int32))
    
    for i in range(num_gpus):
        with cp.cuda.Device(i) as dev:
            dev.synchronize()

    return global_array, cupy_list_A, cupy_list_B

def get_size_info(array_list):
    """Return size of each array and accumulated sizes
    [0, len(a0), len(a1), ...]
    [0, len(a0), len(a0) + len(a1), ...]
    """
    sizes = np.zeros(len(array_list) + 1, dtype=np.uint32)
    for i, array in enumerate(array_list):
        sizes[i + 1] = len(array)
    return sizes, np.cumsum(sizes)

def partition_kernel(A, B, comp, pivot):
    comp[:] = (A < pivot)
    mid = int(comp.sum())
    B[:mid] = A[comp]
    B[mid:] = A[~comp]
    return mid

def partition(arrays: list, pivot, *, workspace: Optional[list] = None, prepend_zero=False):
    """Partition A against pivot and output to B, return mid index."""
    n_partitions = len(arrays)
    _z = 1 if prepend_zero else 0  # int(prepend_zero)
    mid = np.zeros(_z + n_partitions, dtype=np.uint32)

    for i, array_in in enumerate(arrays):
        with array_in.device:
            array_out = workspace[i] if workspace else cp.empty_like(array_in)
            comp = cp.empty_like(array_in, dtype=cp.bool_)
            mid[_z + i] = partition_kernel(array_in, array_out, comp, pivot)
            if not workspace: array_in[:] = array_out[:]
    return mid

def balance_partition(A, left_counts):
    sizes = np.zeros(len(A), dtype=np.uint32)
    for i, array in enumerate(A):
        sizes[i] = len(array)

    remaining_left = np.copy(left_counts)
    remaining_right = np.copy(sizes) - left_counts
    free = np.copy(sizes)

    source_start_left = np.zeros((len(A), len(A)), dtype=np.uint32)
    target_start_left = np.zeros((len(A), len(A)), dtype=np.uint32)
    sz_left = np.zeros((len(A), len(A)), dtype=np.uint32)

    source_start_right = np.zeros((len(A), len(A)), dtype=np.uint32)
    target_start_right = np.zeros((len(A), len(A)), dtype=np.uint32)
    sz_right = np.zeros((len(A), len(A)), dtype=np.uint32)

    # Pack all left data to the left first
    target_idx = 0
    local_target_start = 0

    for source_idx in range(len(A)):
        local_source_start = 0
        message_size = remaining_left[source_idx]
        while message_size > 0:
            max_message = min(free[target_idx], message_size)

            if max_message == 0:
                target_idx += 1
                local_target_start = 0
                continue

            free[target_idx] -= max_message
            remaining_left[source_idx] -= max_message

            sz_left[source_idx, target_idx] = max_message
            source_start_left[source_idx, target_idx] = local_source_start
            target_start_left[source_idx, target_idx] = local_target_start
            local_source_start += max_message
            local_target_start += max_message

            message_size = remaining_left[source_idx]

    # Pack all right data to the right
    for source_idx in range(len(A)):
        local_source_start = left_counts[source_idx]
        message_size = remaining_right[source_idx]
        while message_size > 0:
            max_message = min(free[target_idx], message_size)

            if max_message == 0:
                target_idx += 1
                local_target_start = 0
                continue

            free[target_idx] -= max_message
            remaining_right[source_idx] -= max_message

            sz_right[source_idx, target_idx] = max_message
            source_start_right[source_idx, target_idx] = local_source_start
            target_start_right[source_idx, target_idx] = local_target_start
            local_source_start += max_message
            local_target_start += max_message

            message_size = remaining_right[source_idx]

    return (source_start_left, target_start_left, sz_left), (
        source_start_right,
        target_start_right,
        sz_right,
    )

def scatter(A, B, left_info, right_info):

    source_starts, target_starts, sizes = left_info

    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with cp.cuda.Device(0):
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                # print(source_idx, target_idx, (source_start,
                #      source_end), (target_start, target_end))

                # A[target_idx].array[target_start:target_end] = B[source_idx].array[source_start:source_end]
                target = A[target_idx]
                source = B[source_idx]
                # print("TARGET: ", target, type(target))
                # print("SOURCE: ", source, type(source))
                target[target_start:target_end] = source[source_start:source_end]

    source_starts, target_starts, sizes = right_info
    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with cp.cuda.Device(0):
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                # print(source_idx, target_idx, (source_start,
                #      source_end), (target_start, target_end))

                # A[target_idx].array[target_start:target_end] = B[source_idx].array[source_start:source_end]
                target = A[target_idx]
                source = B[source_idx]
                target[target_start:target_end] = source[source_start:source_end]
