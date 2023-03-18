parser.add_argument("-dev_config", type=str)
args = parser.parse_args()
num_gpus = 4

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

def partition(pivot, beg, end, array, result_array):
    """
    Compare elements in an array with a pivot. 
    If an element is smaller than a pivot, set True
    to a result_array. Otherwise, set False.
    This is not a task, but a normal kernel called by a task.
    This kernel is called by a single GPU.
    """
    for it in range(beg, end):
        if array[it] <= pivot:
            result_array[it] = True
        else:
            # TODO(hc): it may had been initialized to 0
            result_array[it] = False

def scatter_and_merge(input_array, output_array):
    # This function should do the following tasks:
    # First, scatter output crosspys to each device.
    # Second, materialize boolean output arrays and
    #         merge them? TODO(hc): I have no idea!!!!
    # Third, swap the last pivot with the first element in the
    # R array.
    # Then?   
    count = 0
    for d in range(num_gpus):
        with locals.Device[d]:
            true_idices, true_counts, false_idices, false_counts += counts(output_array[d])
        locals.synchronized()

    # Merge elements of true indices as the new crosspy?
    read_idx = []
    read_size = []
    # Merge and partition left side.
    for d in range(num_gpus):
        d_beg, d_end = xp_block_partition(0, true_counts, d, num_gpus)
        d_size = d_end - d_beg + 1
        read_idx.append(d_beg)
        if d == num_gpus - 1:
            read_idx.append(d_end)
        read_size.append(d_size)
        output_cp = cp.zeros((d_size,))
    L_xp = xp.array(read_idx, read_size, array)
    # Set input_array[true_indices] to L_xp.
    fill_array(L_xp, input_array, true_idices)

    # Partition right side.
    for d in range(num_gpus):
        d_beg, d_end = xp_block_partition(0, false_counts, d, num_gpus)
        d_size = d_end - d_beg + 1
        read_idx.append(d_beg)
        if d == num_gpus - 1:
            read_idx.append(d_end)
        read_size.append(d_size)
        output_cp = cp.zeros((d_size,))
    R_xp = xp.array(read_idx, read_size, array)
    # Set input_array[false_indices] to R_xp.
    fill_array(R_xp, input_array, false_idices)

    # Swap pivot and the first element of the right array.
    # So pivot's index is done .
    (input_array[false_indices[0] + offset], input_array[pivot + offset])
        = (input_array[pivot + offset], input_array[false_idices[0] + offset])
    # The above assignment is a lazy operation and so we need synchronization.
    input_array.synchronize()
    return L_xp, R_xp
        
def quick_sort_main(beg, end, array, offset):
    if beg < end:
        pivot = # calculate a pivot
        read_idx = []
        read_size = []
        output_cp_list = []
        for d in range(num_gpus):
            d_beg, d_end = xp_block_partition(beg, end, d, num_gpus)
            d_size = d_end - d_beg + 1
            read_idx.append(d_beg)
            if d == num_gpus - 1:
                read_idx.append(d_end)
            read_size.append(d_size)
            # This declares an output of the partition().
            # If it is True, the corresponding input element has a value
            # greater than a pivot.
            output_cp = cupy.zeros((d_size,), dtype=bool)
            output_cp_list.append(output_cp)
        # Create L and R which contains an array for each device.
        # The input array is partitioned based on read_idx and read_size to gpus.
        array_xp = xp.array(read_idx, read_size, array)
        # TODO(hc): Is there any way to distribute a cupy list to each GPU? 
        # TODO(hc): reuse this output array.
        output_xp = xp.array(output_cp_list)
  
        # This function uses four CUDA devices.
        @spawn(T, placement=[(cuda, cuda, cuda, cuda)], inout=[array_xp])
        def partition_task():
            for d in range(num_gpus):
                with locals.Device[d]:
                    # TODO(hc): Is this necessary? We already partition a crosspy
                    # based on read index and size.
                    d_array_xp = do_read(array_xp[readidx[readsize[d]]])
                    # TODO(hc): How to get output_xd's gpu d's array?
                    d_ouput_xp = do_read(output_xp[d])
                    partition(pivot, 0, len(d_array_xp), d_array_xp, d_output_xp)
        await T # TODO(hc): do/will we support this? 
        L, R = scatter_and_merge(array, output_cp_list, offset)
        # So, size of (L + R) is (input_array size - 1)
        quick_sort_main(L.beg, L.end, L, offset)
        quick_sort_main(R.beg, R.end, R, offset + len(L))

def main(T):
    input_arr = # input array
    quick_sort_main(0, len(input_arr), input_arr)

if __name__ == "__main__":
    with Parla(dev_config_file=args.dev_config):
        T = TaskSpace("T")
        main(T)
