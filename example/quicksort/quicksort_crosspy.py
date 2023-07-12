import math

import numpy as np
import cupy as cp
import crosspy as xp

np.random.seed(10)
cp.random.seed(10)

from common import create_array, partition, get_size_info

fancy_indexing = False

# TODO(wlr): Fuse this kernel, pack this better?
def scatter(active_array, workspace, mid):

    n_partitions = active_array.nparts

    sizes, size_prefix = get_size_info(active_array.device_array.values())
    left_prefix = np.cumsum(mid)
    right_prefix = size_prefix - left_prefix
    print("Size Prefix: ", size_prefix)
    print("Left Prefix: ", left_prefix)
    print("Right Prefix: ", right_prefix)

    global_left_count = np.sum(mid)
    global_right_count = len(active_array) - global_left_count

    if isinstance(workspace, list):
        if workspace:
            left_array, right_array = workspace
        else:
            # Create new crosspy arrays for the left and right partitions
            print("Number of elements in the left partition: ", global_left_count)
            num_left_blocks = math.ceil(global_left_count / args.m)
            print("Number of left blocks: ", num_left_blocks)

            # Allocate left array
            left_cupy_blocks = []
            for i in range(num_left_blocks):
                local_length = (int)(min(args.m, global_left_count - i*args.m))

                with cp.cuda.Device(0):
                    left_cupy_blocks.append(cp.zeros(local_length, dtype=cp.int32))

            left_array = xp.array(left_cupy_blocks, axis=0)
            workspace.append(left_array)

            print("Number of elements in the right partition: ", global_right_count)
            num_right_blocks = (int)(math.ceil(global_right_count / args.m))
            print("Number of right blocks: ", num_right_blocks)

            # Allocate right array
            right_cupy_blocks = []
            for i in range(num_right_blocks):
                local_length = (int)(min(args.m, global_right_count - i*args.m))

                with cp.cuda.Device(0):
                    right_cupy_blocks.append(cp.zeros(local_length, dtype=cp.int32))

            right_array = xp.array(right_cupy_blocks, axis=0)
            workspace.append(right_array)

            print("Left array: ", left_array)
            print("Left array 0: ", left_array[0:2])
            print("Right array: ", right_array)
            print("Right array 0: ", right_array[0:2])
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
        print("Left send indices: ", left_indexing)
        print("Right send indices: ", right_indexing)

        if isinstance(workspace, list):
            # Performing left scatter
            left_array[:] = active_array[left_indexing]
            # Performing right scatter
            right_array[:] = active_array[right_indexing]
        else:
            active_array[:] = workspace[np.concatenate((left_indexing, right_indexing))]
    else:
        for i, array in enumerate(workspace.block_view()):
            # Write left
            print("left", left_prefix[i], left_prefix[i+1], mid[i+1])
            if mid[i+1] > 0:
                print("A", active_array[left_prefix[i]:left_prefix[i+1]],
                    len(active_array[left_prefix[i]:left_prefix[i+1]]))
                print("B local", array[:mid[i+1]], len(array[:mid[i+1]]))
                print("B global", workspace[size_prefix[i]:size_prefix[i+1]+mid[i+1]],
                    len(workspace[size_prefix[i]:size_prefix[i+1]+mid[i+1]]))

                # QUESTION: How can I perform this copy?
                # xA[left_prefix[i]:left_prefix[i+1]] = array[:mid[i+1]]
                active_array[left_prefix[i]:left_prefix[i+1]
                ] = workspace[size_prefix[i]:size_prefix[i]+mid[i+1]]
            # Write right
            print("right", right_prefix[i], right_prefix[i+1])

            if (sizes[i+1] - mid[i+1]) > 0:
                print("A", active_array[right_prefix[i]:right_prefix[i+1]],
                    len(active_array[right_prefix[i]:right_prefix[i+1]]))
                print("B local", array[mid[i+1]:], len(array[mid[i+1]:]))
                print("B global", workspace[size_prefix[i]+mid[i+1]:size_prefix[i+1]],
                    len(workspace[size_prefix[i]+mid[i+1]:size_prefix[i+1]]))

                # QUESTION: How can I perform this copy?
                # xA[right_prefix[i]:right_prefix[i+1]] = array[mid[i+1]:]
                active_array[right_prefix[i]:right_prefix[i+1]
                ] = workspace[size_prefix[i]+mid[i+1]:size_prefix[i+1]]

        if global_left_count > 0:
            print("Array left", active_array[:global_left_count])
        if global_right_count > 0:
            print("Array right", active_array[global_left_count:])


def quicksort(active_array, workspace=None):
    print("----------------------")
    print("Active array: ", active_array)
    n_partitions = active_array.nparts

    print("CrossPy has n_partitions: ", n_partitions)

    if len(active_array) <= 1:
        print("Base case reached, returning...")
        return

    pivot_idx = np.random.randint(0, len(active_array))
    pivot_idx = -1
    pivot_idx = len(active_array) - 1

    print("Active partition has shape: ", active_array.shape)
    print("Active partition has len: ", len(active_array))

    print("The chosen pivot index is: ", pivot_idx)

    pivot = (int)(active_array[pivot_idx])
    print("The chosen pivot is: ", pivot)

    # local partition
    print("Performing local partition...")
    mid = partition(list(active_array.block_view()), pivot, workspace=list(workspace.block_view()) if workspace else [], prepend_zero=True)
    print("Found the following splits: ", mid)


    # Scatter to other partitions
    print("Performing local scatter...")
    scatter(active_array, workspace or [], mid)

    global_left_count = np.sum(mid)
    quicksort(active_array[:global_left_count], workspace)
    quicksort(active_array[global_left_count:], workspace)

    # print("Starting Partition on Slice: ", active_slice)

    # # form slices to pass to children
    # previous_start = active_slice.start
    # previous_end = active_slice.stop

    # left_start = (int)(previous_start)
    # left_end = (int)(previous_start + local_split)
    # left_slice = slice(left_start, left_end)

    # right_start = (int)(previous_start + local_split)
    # right_end = (int)(previous_end)
    # right_slice = slice(right_start, right_end)

    # quicksort(global_array, left_array, left_slice, T)
    # quicksort(global_array, right_array, right_slice, T)

def main(args):
    global_array, cupy_list_A, cupy_list_B = create_array(args.m, args.num_gpus)

    xA = xp.array(cupy_list_A, axis=0)
    xB = xp.array(cupy_list_B, axis=0)

    quicksort(xA, xB)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_gpus", type=int, default=2)
    parser.add_argument("-m", type=int, default=5)
    args = parser.parse_args()
    main(args)
