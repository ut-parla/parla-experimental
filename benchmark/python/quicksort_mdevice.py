import argparse

from parla import Parla, spawn, TaskSpace

parser = argparse.ArgumentParser()
parser.add_argument("-dev_config", type=str)
args = parser.parse_args()

def xp_block_partition(start, end, dev_id, num_gpus):
    """
    This function partitions a range by a device id
    and returns a disjointed range in a [beg, end) form.
    """
    xp_len = (end - start) + 1
    block_size = int(xp_len / num_gpus)
    block_beg = dev_id * block_size
    block_end = block_beg + block_size
    remaining = xp_len % num_gpus
    if remaining > 0:
        if dev_id >= remaining:
            block_beg += remaining
            block_end += remaining
        else:
            block_beg += dev_id
            block_end += (dev_id + 1)
    return block_beg, block_end

"""
def partition():
    pass
"""

def main(T):
    device_id = 0
    num_gpus = 4
    num_arrays = 1
    @spawn(T, placement=[(cuda, cuda, cuda, cuda)])
    def quicksort():
        while True:
            remaining_work = False
            target_lists = []
            if num_arrays == 1: # which means that the first level.
                target_lists.append(input_array)
            for i in range(num_arrays)
                target = target_lists[i]
                beg = 0
                end = len(target)
                # pivot = find_pivot(beg, end)
                readidx = []
                readsize = []
                for d in range(num_gpus):
                    d_beg, d_end = xp_block_partition(beg, end, d, num_gpus)
                    readidx.append(d_beg)
                    readsize.append(d_end - d_beg + 1)
                    if (d == num_gpus - 1):
                        readidx.append(d_end)
                # x = xp.array(range(end), readidx, readsize)
                for d in range(num_gpus):
                    """
                    with locals.Device[d]:
                          # Iterates the ranges, and stores items smaller
                          # than a pivot to low_arr and items bigger than
                          # a pivot to high_arr
                          partition(d_beg, d_end, pivot, low_arr[d], high_arr[d])
                          # TODO(hc): How can we create an array of a vector?
                          #           We don't know size of low_arr[d] and high_arr[d].
                    """
                    pass
                for d in range(num_gpus):
                    """
                    with locals.Device[d]:
                          # Aggregate low_arr and high_arr from the whole
                          # devices and merge them to a single array.
                          # For each level computation (e.g., ith level),
                          # i^2 arrays are created.
                          # For example, after the first level computation,
                          # left and right arrays are created and each of them
                          # should be sorted separately. 

                          # Let's assume that this loop's outputs are L and R

                          # if L or R is not sorted:
                          #    remaining_work = True
                    """
                    pass
                if remaining_work == False:
                    break
                target_lists.append(L)
                target_lists.append(R)
            num_arrays *= 2


if __name__ == "__main__":
    with Parla(dev_config_file=args.dev_config):
        T = TaskSpace("T")
        main(T)
