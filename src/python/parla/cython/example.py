
@spawn(placement=device, IN=[datablocks])
def task():
    pass


@data((GPU0, GPU1), 0, IN=[A])
@data((GPU0, GPU1), 1, IN=[B])
@spawn()
def task():
    pass


@data((GPU0, GPU1), 0=[A], 1=[B]) @ defaults to inout
@spawn(IN=[A])
def task():
    pass


A = library_call()
dist_array = CrossPy(A, placement=[GPU0], wrapper=parray)
placement = [GPU0, GPU0, GP]


device_set = parla.get_all_gpus()

partitions = Partitions()
partitions.add({0: idx})
partitions.add({0: idx1, 1: idx2})
partitions.add({0: idx1, 1: idx2, 2: idx1, 3: idx2})

dist_array = CrossPy(array,
                     coloring=partitions,
                     device_set=device_set,
                     wrapper=parla
                     )

# Run on 1, 2, or 4 GPUs


@spawn(placement=[GPU, (GPU, GPU), (GPU, GPU, GPU, GPU)], IN=dist_array)
def task():
    multi_device_kernel(dist_array)
