import numpy as np
import cupy as cp

np.random.seed(10)
cp.random.seed(10)

import math

def slice_distributed_arrays(dist_arrays, size_prefix, start=None, end=None, *, workspace=None):
    # print("START: ", start, "END: ", end)
    assert start is None or end is None or start <= end

    # print("GLOBAL_PREFIX: ", global_prefix)
    if start is not None:
        start = int(start)
        global_start_idx = np.searchsorted(size_prefix, start, side="right") - 1
        local_left_split = int(start - size_prefix[global_start_idx])
        assert local_left_split < len(dist_arrays[global_start_idx])
    else:
        global_start_idx = 0
        local_left_split = None

    if end is not None:
        end = int(end)
        global_end_idx = np.searchsorted(size_prefix, end, side="right") - 1
        local_right_split = int(end - size_prefix[global_end_idx])
    else:
        global_end_idx = len(dist_arrays)
        local_right_split = None

    # print("GLOBAL_START: ", global_start_idx, "GLOBAL_END: ", global_end_idx)
    # print("LOCAL_START: ", local_left_split, "LOCAL_END: ", local_right_split)

    sliced_arrays = []
    sliced_workspace = []

    # Reform a global array out of sliced components (NOTE: THESE SEMANTICS BREAK PARRAY. Concurrent slices cannot be written)
    if global_start_idx == global_end_idx:
        sliced_arrays.append(dist_arrays[global_start_idx][local_left_split:local_right_split])
        if workspace is not None:
            sliced_workspace.append(workspace[global_start_idx][local_left_split:local_right_split])
            return sliced_arrays, sliced_workspace
        return sliced_arrays

    assert global_start_idx < global_end_idx
    sliced_arrays.append(dist_arrays[global_start_idx][local_left_split:])
    if workspace is not None:
        sliced_workspace.append(workspace[global_start_idx][local_left_split:])

    for i in range(global_start_idx + 1, global_end_idx):
        sliced_arrays.append(dist_arrays[i])
        if workspace is not None:
            sliced_workspace.append(workspace[i])

    if global_end_idx < len(dist_arrays) and local_right_split > 0:
        sliced_arrays.append(dist_arrays[global_end_idx][:local_right_split])
        if workspace is not None:
            sliced_workspace.append(workspace[global_end_idx][:local_right_split])

    if workspace is not None:
        return sliced_arrays, sliced_workspace
    return sliced_arrays

def partitioned_zeros_by_size(global_size, block_size, devices=None):
    # Create new arrays for the left and right partitions
    # print("Number of elements in the partition: ", global_count)
    num_blocks = math.ceil(global_size / block_size)
    # print("Number of blocks: ", num_blocks)

    # Allocate array
    cupy_blocks = []
    for i in range(num_blocks):
        local_length = (int)(min(block_size, global_size - i*block_size))
        with devices[i] if devices else cp.cuda.Device(0):
            cupy_blocks.append(cp.zeros(local_length, dtype=cp.int32))
    return cupy_blocks

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

def cupy_copy(source, source_start, source_end, target, target_start, target_end, *, src_device):
    with src_device:
        temp = cp.ascontiguousarray(source[source_start:source_end])
    target[target_start:target_end] = cp.asarray(temp)

def scatter(A, B, left_info, right_info, ctx_func, copy_func):
    source_starts, target_starts, sizes = left_info

    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with ctx_func(target_idx):
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                target = A[target_idx]
                source = B[source_idx]
                copy_func(source, source_start, source_end, target, target_start, target_end, src_device=ctx_func(source_idx))

    source_starts, target_starts, sizes = right_info
    for source_idx in range(len(A)):
        for target_idx in range(len(A)):
            if sizes[source_idx, target_idx] == 0:
                continue

            with ctx_func(target_idx):
                source_start = source_starts[source_idx, target_idx]
                source_end = source_start + sizes[source_idx, target_idx]

                target_start = target_starts[source_idx, target_idx]
                target_end = target_start + sizes[source_idx, target_idx]

                target = A[target_idx]
                source = B[source_idx]

                copy_func(source, source_start, source_end, target, target_start, target_end, src_device=ctx_func(source_idx))
