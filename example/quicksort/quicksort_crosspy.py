import numpy as np
import cupy as cp
import crosspy as xp

np.random.seed(10)
cp.random.seed(10)

import time
from common import create_array, partition, get_size_info
from cupy_helper import partitioned_zeros_by_size

fancy_indexing = False

# TODO(wlr): Fuse this kernel, pack this better?
def scatter(active_array, workspace, mid):

    n_partitions = active_array.nparts

    sizes, size_prefix = get_size_info(active_array.device_array.values())
    left_prefix = np.cumsum(mid)
    right_prefix = size_prefix - left_prefix
    # print("Size Prefix: ", size_prefix)
    # print("Left Prefix: ", left_prefix)
    # print("Right Prefix: ", right_prefix)

    global_left_count = np.sum(mid)
    global_right_count = len(active_array) - global_left_count

    if isinstance(workspace, list):
        if workspace:
            left_array, right_array = workspace
        else:
            left_cupy_blocks = partitioned_zeros_by_size(global_left_count, args.m)
            right_cupy_blocks = partitioned_zeros_by_size(global_right_count, args.m)

            left_array = xp.array(left_cupy_blocks, axis=0 if len(left_cupy_blocks) else None)
            right_array = xp.array(right_cupy_blocks, axis=0 if len(right_cupy_blocks) else None)
            workspace += [left_array, right_array]
            # print("Left array: ", left_array)
            # print("Left array 0: ", left_array[0:2])
            # print("Right array: ", right_array)
            # print("Right array 0: ", right_array[0:2])
    else:
        right_prefix += global_left_count

    if fancy_indexing:
        left_indexing = np.zeros(global_left_count, dtype=np.uint32)
        right_indexing = np.zeros(global_right_count, dtype=np.uint32)
        for i in range(n_partitions):
            left_indexing[left_prefix[i]:left_prefix[i+1]
                            ] = np.arange(size_prefix[i], size_prefix[i]+mid[i+1])
            right_indexing[right_prefix[i]:right_prefix[i+1]
                            ] = np.arange(size_prefix[i]+mid[i+1], size_prefix[i+1])
        # print("Left send indices: ", left_indexing)
        # print("Right send indices: ", right_indexing)

        if isinstance(workspace, list):
            # Performing left scatter
            left_array[:] = active_array[left_indexing]
            # Performing right scatter
            right_array[:] = active_array[right_indexing]
        else:
            active_array[:] = workspace[np.concatenate((left_indexing, right_indexing))]
    else:
        for i, array in enumerate(n_partitions):
            if isinstance(workspace, list):
                if mid[i+1] > 0:
                    left_array[left_prefix[i]:left_prefix[i+1]
                        ] = workspace[size_prefix[i]:size_prefix[i]+mid[i+1]]
                if right_prefix[i+1] - right_prefix[i] > 0:
                    right_array[right_prefix[i]:right_prefix[i+1]
                        ] = workspace[size_prefix[i]+mid[i+1]:size_prefix[i+1]]
            else:
                # Write left
                # print("left", left_prefix[i], left_prefix[i+1], mid[i+1])
                if mid[i+1] > 0:
                    # print("A", active_array[left_prefix[i]:left_prefix[i+1]],
                    #     len(active_array[left_prefix[i]:left_prefix[i+1]]))
                    # print("B local", array[:mid[i+1]], len(array[:mid[i+1]]))
                    # print("B global", workspace[size_prefix[i]:size_prefix[i+1]+mid[i+1]],
                    #     len(workspace[size_prefix[i]:size_prefix[i+1]+mid[i+1]]))

                    # QUESTION: How can I perform this copy?
                    # xA[left_prefix[i]:left_prefix[i+1]] = array[:mid[i+1]]
                    active_array[left_prefix[i]:left_prefix[i+1]
                    ] = workspace[size_prefix[i]:size_prefix[i]+mid[i+1]]
                # Write right
                # print("right", right_prefix[i], right_prefix[i+1])

                if right_prefix[i+1] - right_prefix[i] > 0:
                    # print("A", active_array[right_prefix[i]:right_prefix[i+1]],
                    #     len(active_array[right_prefix[i]:right_prefix[i+1]]))
                    # print("B local", array[mid[i+1]:], len(array[mid[i+1]:]))
                    # print("B global", workspace[size_prefix[i]+mid[i+1]:size_prefix[i+1]],
                    #     len(workspace[size_prefix[i]+mid[i+1]:size_prefix[i+1]]))

                    # QUESTION: How can I perform this copy?
                    # xA[right_prefix[i]:right_prefix[i+1]] = array[mid[i+1]:]
                    active_array[right_prefix[i]:right_prefix[i+1]
                    ] = workspace[size_prefix[i]+mid[i+1]:size_prefix[i+1]]

        # if global_left_count > 0:
        #     print("Array left", active_array[:global_left_count])
        # if global_right_count > 0:
        #     print("Array right", active_array[global_left_count:])


def quicksort(array, workspace=None):
    # print("----------------------")
    # print("Active array: ", active_array)
    # print("CrossPy has n_partitions: ", active_array.nparts)

    if len(array) < 2:
        # print("Base case reached, returning...")
        return

    pivot_idx = np.random.randint(0, len(array))
    pivot_idx = -1
    pivot_idx = len(array) - 1

    # print("Active partition has shape: ", active_array.shape)
    # print("Active partition has len: ", len(active_array))

    # print("The chosen pivot index is: ", pivot_idx)

    pivot = (int)(array[pivot_idx])
    # print("The chosen pivot is: ", pivot)

    # local partition
    # print("Performing local partition...")
    mid = partition(list(array.block_view()), pivot, workspace=list(workspace.block_view()) if workspace else [], prepend_zero=True)
    # print("Found the following splits: ", mid)


    # Scatter to other partitions
    # print("Performing local scatter...")
    scatter(array, workspace if workspace is not None else [], mid)

    global_left_count = np.sum(mid)
    quicksort(array[:global_left_count], workspace[:global_left_count] if workspace is not None else None)
    quicksort(array[global_left_count:], workspace[global_left_count:] if workspace is not None else None)

def main(args):
    global_array, dist_arrays, workspace_list = create_array(args.m, args.num_gpus, unique=True)

    x = xp.array(dist_arrays, axis=0)
    xWorkspace = xp.array(workspace_list, axis=0)

    print("Original Array: ", x, flush=True)

    t_start = time.perf_counter()
    quicksort(x, xWorkspace)
    t_end = time.perf_counter()

    print("Sorted:")
    print(x)
    print("Time: ", t_end - t_start)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_gpus", type=int, default=2)
    parser.add_argument("-m", type=int, default=5, help="Per device size")
    args = parser.parse_args()
    main(args)
