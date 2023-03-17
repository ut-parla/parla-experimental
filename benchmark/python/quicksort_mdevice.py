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

def main(T):
    device_id = 0
    for i in range(4):
        beg, end = xp_block_partition(0, 102, i, 4)
        print(i, "-> ", beg, "~", end)

    """
    @spawn(T[0])
    def quick_sort():
        print("Hello World")

        # Choose pivot
        # 
        # with cp.cuda.Device(device_id):
        #   partition()
        # device_id += 1
        # 
        # quick_sort()
        # quick_sort()
    """

if __name__ == "__main__":
    with Parla(dev_config_file=args.dev_config):
        T = TaskSpace("T")
        main(T)
