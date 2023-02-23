
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


device_set = parla.get_all_gpus()

partitions = Partitions()
partitions.add({0: idx1, 1: idx2})
partitions.add({0: idx1, 1: idx2, 2: idx3, 3: idx4})

dist_array = CrossPy(array,
                     coloring=partitions,
                     device_set=device_set,
                     wrapper=parla
                     )

# Run on 1, 2, or 4 GPUs


@spawn(placement=[GPU, (GPU, GPU), (GPU, GPU, GPU, GPU)], IN=dist_array)
def task():
    multi_device_kernel(dist_array)


  10:35 AM
I think that makes sense. It certainly makes much more sense than my original hack to over-partition at array creation for all possible uses within a task. There's a lot of behind the scenes decisions to make about how the scattering is done and tracked, but we can start with a simple lazy redistribution when the task runs. (edited) 
  11:05 AM
One thing I was trying (poorly) to explain was that multiple colorings depending on the task environment could also be a property of the distributed array, instead of a property of the task.

device_set = parla.get_all_gpus()

partitions = Partitions()
partitions.add({0: idx1, 1: idx2})
partitions.add({0: idx1, 1: idx2, 2: idx3, 3: idx4})

dist_array = CrossPy(array,
                     coloring=partitions,
                     device_set=device_set,
                     wrapper=parla
                     )

# Run on 1, 2, or 4 GPUs. Partition is automatically redistributed. 
@spawn(placement=[GPU, (GPU, GPU), (GPU, GPU, GPU, GPU)], IN=dist_array)
def task():
    multi_device_kernel(dist_array)

(edited)
11:05
To separate the logical partitioning (coloring) from the device set it is actualized on (device_set) (edited) 
11:10
This would/could allow for abstract partitioners without user specification:

partition = Partitions(default=1D_Equal)

dist_array = CrossPy(array,
                     coloring=partitions,
                     device_set=device_set,
                     wrapper=parla
                     )

To handle all device sets with some default behavior. (edited) 