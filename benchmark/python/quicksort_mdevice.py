parser.add_argument("-dev_config", type=str)
args = parser.parse_args()
num_gpus = 4

# TODO(hc): numba to make this run on gpu.
def fill_array(input_array, offset, values):
    # In this case, input_array[i + offset] could be locating in
    # a different gpu from the current device who is running this code.
    # But, xparray automatically handles and updates values
    # in a lazy manner (or we can call synchronize() to immedaitely reflect
    # this)
    for i in range(len(values)):
        input_array[i + offset] = values[i]

# TODO(hc): numba to make this run on gpu.
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

# TODO(hc): numba to make this run on gpu.
def partition(pivot, array, result_array):
    """
    Compare elements in an array with a pivot. 
    If an element is smaller than a pivot, set True
    to a result_array. Otherwise, set False.
    This is not a task, but a normal kernel called by a task.
    This kernel is called by a single GPU.
    """
    for it in range(len(array)):
        if array[it] <= pivot:
            result_array[it] = True
        else:
            # TODO(hc): it may had been initialized to 0
            result_array[it] = False

# TODO(hc): numba to make this run on gpu.
def scatter_and_merge(input_array, output_array, offset, pivot_idx):
    # This function should do the following tasks:
    # First, scatter output crosspys to each device.
    # Second, materialize boolean output arrays and
    #         merge them? TODO(hc): I have no idea!!!!
    # Third, swap the last pivot with the first element in the
    # R array.
    # Then?   
    count = 0
    # To store prefix sum
    left_prefix_sum = xp.array([cp.zeros(1,)] * num_gpus)
    right_prefix_sum = xp.array([cp.zeros(1,)] * num_gpus)
    # The first dimension is a device id and the second dimension
    # stores left/rigth values based on a pivot.
    left_value_array = xp.array("""How to initilaize to store a list of indices?""")
    right_value_array = xp.array("""TODO(hc)..""")
    for d in range(num_gpus):
        with locals.Device[d]:
            # Aggregate actual values of the left and the right array elements.
            # Note that left values and right values are newly allocated arrays
            # and contains copied original values.
            # e.g., if output_array[0][1] == True:
            #           left_values.append(input_array[d][offset + 1])?
            left_values, left_counts, right_values, right_counts += counts(
                output_array[d], input_array, offset, d)
            left_prefix_sum[d] = left_counts
            right_prefix_sum[d] = right_counts
            # Left/right values are extracted and copied from
            # the input array, but in this case, just store references.
            left_value_array[d] = left_values
            right_value_array[d] = right_values
        # Crosspy's default model is a lazy write.
        # To flush requested operations immediately, call synchronize.
        locals.synchronize()

    # Sequentially calculates a prefix sum.
    for d in range(0, num_gpus - 1):
        with locals.Device[d]:
            left_prefix_sum[d + 1] += left_prefix_sum[d]
            right_prefix_sum[d + 1] += right_prefix_sum[d]
    left_prefix_sum.synchronize()
    right_prefix_sum.synchronize()

    # Sequentially fill left and right of the original inputs using prefix sum.
    # TODO(hc): conisder offset!
    first_right_index = 0
    for d in range(num_gpus):
        with locals.Device[d]:
            l_offset = 0 if d == 0 else left_prefix_sum[d - 1]
            l_end = left_prefix_sum[d]
            r_offset = 0 if d == 0 else right_prefix_sum[d - 1]
            r_end = right_prefix_sum[d]
            # Now, we replace the original input arrays with new left/right values. 
            # First, input_array[l_offset + offset:l_end + offset] <- left_values[:]
            # Second, input_array[r_offset + offset:r_end + offset] <- right_values[:]
            #fill_array(input_array, l_offset + offset, l_end + offset, left_values)
            #fill_array(input_array, r_offset + offset, r_end + offset, right_values) 

            # TODO(hc) If xparray supports a local index for a slice of the xparray, then
            # we can use the following pattern. Otherwise, we should do like the above.

            fill_array(input_array, l_offset, left_values)
            fill_array(input_array, r_offset, right_values)
              
            if d == 0:
                #first_right_index = r_offset + offset
                # TODO(hc) If xparray spports a local index of a slice of the xparray
                # then we can use the following pattern.
                first_right_index = r_offset + offset

    # TODO(hc): who call this?
    # Swap pivot and the first element of the right array.
    # So pivot's index is done, which means that it will not be accessed again.
    (input_array[first_right_index], input_array[pivot_idx])
        = (input_array[pivot_idx], input_array[first_right_index])
    # The above assignment is a lazy operation and so we need synchronization.
    input_array.synchronize()
    # Return the final position of the pivot.
    return first_right_index 
        
# This should be a function since we need to pass parameters.
def quick_sort_main(beg, end, array, offset, num_gpus):
    if beg < end:
        # TODO(hc): is there any way to copy an element of the crosspy array
        #           regardless of the current location?
        pivot_idx = end - 1
        pivot = array[pivot_idx]
        read_idx = []
        read_size = []
        output_cp_list = []
        # Partition the input array.
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
            # TODO(hc): This can be reused.
            output_cp = cupy.zeros((d_size,), dtype=bool)
            output_cp_list.append(output_cp)
        # This repartitions a slice of the input array (the original input array)
        # to target devices passed to this function.
        # The input array is partitioned based on read_idx and read_size to gpus.
        # Each device iterates each element in a partition and splits to
        # elements smaller than and equal to a pivot and ones greater than a pivot.
        array_xp = xp.repartition(read_idx, read_size, array)
        # TODO(hc): Is there any way to distribute a cupy list to each GPU? 
        # TODO(hc): allocate once by using the very initial input partition size
        #           and reuse it recursively.
        #           So, always needs zerofication.
        output_xp = xp.array(output_cp_list)

        # Construct placement.
        ps = ()
        for g in range(0, num_gpus):
            ps.append((cuda))
  
        # This function uses four CUDA devices.
        @spawn(T, placement=[ps], inout=[array_xp])
        def partition_task():
            for d in range(num_gpus):
                with locals.Device[d]:
                    # TODO(hc): Is this necessary? We already partition a crosspy
                    # based on read index and size.
                    d_array_xp = do_read(array_xp[readidx[readsize[d]]])
                    # TODO(hc): How to get output_xd's gpu d's array?
                    d_ouput_xp = do_read(output_xp[d])
                    partition(pivot, d_array_xp, d_output_xp)
                new_pivot_idx = scatter_and_merge(array, output_cp_list, offset, pivot_idx)
                # So, size of (L + R) is (input_array size - 1)
                next_num_gpus = num_gpus/2
                # At least, one gpu should be used.
                if next_num_gpus == 0:
                    next_num_gpus = 1
                quick_sort_main(beg, new_pivot_idx - 1, array, offset, next_num_gpus)
                quick_sort_main(new_pivot_idx + 1, end, array, offset + new_pivot_idx, next_num_gpus)

        await T # TODO(hc): do/will we support this? 

def main(T):
    input_arr = # input array
    quick_sort_main(0, len(input_arr), input_array, num_gpus)

if __name__ == "__main__":
    with Parla(dev_config_file=args.dev_config):
        T = TaskSpace("T")
        main(T)
