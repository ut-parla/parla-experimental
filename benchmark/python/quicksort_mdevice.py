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
def block_partition(start, end, dev_id, num_gpus):
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
def partition(pivot, input_array, output_array):
    """
    Compare elements in an array with a pivot. 
    If an element is smaller than a pivot, set True
    to a result_array. Otherwise, set False.
    This is not a task, but a normal kernel called by a task.
    This kernel is called by a single GPU.
    """
    for it in range(len(input_array)):
        # TODO(hc): it may had been initialized to 0
        output_array[input_array[it] <= pivot] = True

# TODO(hc): numba to make this run on gpu.
def counts(output_array, input_array, left_values, right_values):
    # Assume that the sliced arrays support local index.
    right_index = 0
    left_index = 0
    for i in range(len(output_array)):
        if output_array[i] == True:
            right_values[right_index] = input_array[i]
            right_index += 1
        else:
            left_values[left_index] = input_array[i]
            left_index += 1
    return left_index, right_index

# TODO(hc): numba to make this run on gpu.
def scatter_and_merge(input_array, output_array):
    # This function should do the following tasks:
    # First, scatter output crosspys to each device.
    # Third, swap the last pivot with the first element in the
    # R array.
    count = 0
    # To store prefix sum
    left_prefix_sum = np.zeros((num_gpus,))
    right_prefix_sum = np.zeros((num_gpus,))
    # Dense array; so depending on the results, some last elements are not
    # used. The end index is specified through the below counts.
    left_value_array = xp.from_shape(input_array.shape)
    right_value_array = xp.from_shape(input_array.shape)
    for d in range(num_gpus):
        with locals.Device[d]:
            # Aggregate actual values of the left and the right array elements.
            # Those aggregated values are held in right_ and left_value_arrays. 
            left_counts, right_counts += counts(
                output_array.get_partition(d), input_array.get_partition(d),
                left_value_array.get_partition(d), right_value_array.get_partition(d))
            left_prefix_sum[d] = left_counts
            right_prefix_sum[d] = right_counts
        # Crosspy's default model is a lazy write.
        # To flush requested operations immediately, call synchronize.
        locals.synchronize()

    # Sequentially calculates a prefix sum.
    # Run this prefix sum in CPU.
    for d in range(0, num_gpus - 1):
        left_prefix_sum[d + 1] += left_prefix_sum[d]
        right_prefix_sum[d + 1] += right_prefix_sum[d]

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

            # TODO(hc): Does xparray also support operators with cupy or numpy arrays?
            #           (that the lazy update)
            fill_array(input_array, l_offset, left_value_array.get_partition(d))
            fill_array(input_array, r_offset, right_value_array.get_partition(d))
              
            if d == 0:
                #first_right_index = r_offset + offset
                # TODO(hc) If xparray spports a local index of a slice of the xparray
                # then we can use the following pattern.
                first_right_index = r_offset

    # TODO(hc): who call this?
    # Swap pivot and the first element of the right array.
    # So pivot's index is done, which means that it will not be accessed again.
    (input_array[first_right_index], input_array[-1])
        = (input_array[-1], input_array[first_right_index])
    # The above assignment is a lazy operation and so we need synchronization.
    input_array.synchronize()
    # Return the final position of the pivot
    return first_right_index
        
# This should be a function since we need to pass parameters.
def quick_sort_main(target_xparray, beg, end, input_array, offset):
    # Always use the last element as the pivot.
    pivot = target_xparray[-1]

    # Deep copy from the input xp array with its partitioning. 
    # Existing values will be overwritten so fine.
    comparison_output_xp = xp.from_shape(target_xparray.shape, type=bool)

    # Construct placement.
    ps = ()
    # TODO: new xpy interface is necessary
    num_gpus = target_xparray.get_num_partitions()
    for g in range(0, num_gpus):
        ps.append((cuda))
  
    @spawn(T, placement=[ps], inout=[target_xparray])
    def partition_task():
        for d in range(num_gpus):
            with locals.Device[d]:
                # Partition the input with the left and the right partiitons
                # based on the pivot.
                partition(pivot, target_xparray.get_partition(d), comparison_output_xp.get_partition(d))
            new_pivot_idx = scatter_and_merge(target_xparray, comparison_output_xp)

            if left_slice_size > 0:
                # Slice the range from 0 to left_slice_size of target_xparray and create new crosspy array variable.
                # Internally share PArrays between the parent crosspy and this.

                # As the current PArray does not support nested slicing,
                # slice the original input array with its offset.
                left_xparray = input_array[offset : offset+new_pivot_idx-1]
                quick_sort_main(left_xparray, offset, offset+new_pivot_idx-1, input_array, offset)
            if right_slice_size > 0:
                right_xparray = input_array[offset+new_pivot_idx+1:offset+end]
                quick_sort_main(right_xparray, offset+new_pivot_idx+1, offset+end, \
                                input_array, offset)
    await T

def main(T):
    input_arr = # input xp array. first it knows the current number of gpus.
    # We need to pass the original input array since PArray does not support 
    # nested slicing and we should update the original PArray with the offset.
    quick_sort_main(input_array, 0, len(input_array), input_array, 0)

if __name__ == "__main__":
    with Parla(dev_config_file=args.dev_config):
        T = TaskSpace("T")
        main(T)
