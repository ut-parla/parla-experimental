
import argparse
import os
# import nvtx
# import mlk

parser = argparse.ArgumentParser()
#Size of matrix
parser.add_argument('-n', type=int, default=4, help='Size of matrix')
#How many trials to run
parser.add_argument('-trials', type=int, default=1)
#What mapping to use
parser.add_argument('-fixed', type=int, default=1)
#How many gpus to use
parser.add_argument('-ngpus', type=int, default=2)
parser.add_argument('-t', type=int, default=1)
parser.add_argument('-workers', type=int, default=4)


args = parser.parse_args()

load = 1.0/args.workers
# mkl.set_num_threads(args.t)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.t)
os.environ["OMP_NUM_THREADS"] = str(args.t)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.t)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.t)
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = cuda_visible_devices.strip().split(',')
    cuda_visible_devices = list(map(int, cuda_visible_devices))

gpus = cuda_visible_devices[:args.ngpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
ngpus = args.ngpus
#IMPORT CUPY AND PARLA AFTER CUDA_VISIBLE_DEVICES IS SET

import time
import numba
import numpy as np
import cupy as cp
import crosspy as xp
# from crosspy import PartitionScheme
from numba import cuda

import resource
import sys
from parla import Parla, spawn
# from parla.array import copy
from parla.cython.device_manager import cpu, gpu
from parla.cython.variants import specialize
# from parla.ldevice import LDeviceSequenceBlocked
from parla.cython.tasks import TaskSpace, CompletedTaskSpace
import argparse
from parla.cython.tasks import GPUEnvironment
from parla.cython.device import PyGPUDevice
import crosspy.device

@crosspy.device.of(PyGPUDevice)
def converter(parlagpu):
    return xp.gpu(parlagpu.id)

num_tests = args.trials

def stream_cupy_to_numba(cp_stream):
    '''
    Notes:
        1. The lifetime of the returned Numba stream should be as long as the CuPy one,
           which handles the deallocation of the underlying CUDA stream.
        2. The returned Numba stream is assumed to live in the same CUDA context as the
           CuPy one.
        3. The implementation here closely follows that of cuda.stream() in Numba.
    '''
    from ctypes import c_void_p
    import weakref

    # get the pointer to actual CUDA stream
    raw_str = cp_stream.ptr

    # gather necessary ingredients
    ctx = cuda.devices.get_context()
    handle = c_void_p(raw_str)
    finalizer = None  # let CuPy handle its lifetime, not Numba

    # create a Numba stream
    nb_stream = cuda.cudadrv.driver.Stream(weakref.proxy(ctx), handle, finalizer)

    return nb_stream

def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return "%s: usertime=%s systime=%s mem=%s mb \
           "%(point, usage[0], usage[1],
                usage[2] / 1024.0 )
@specialize
@numba.jit(parallel=True)
def jacobi(j, part_size, block_size, a0, a1):
    """
    CPU code to perform a single step in the Jacobi iteration.
    """
    
    # if(len(r) < len(a1)):
    #     r = np.append(r, np.zeros((len(a1) - len(r),), dtype=int))
    # if(len(l) < len(a1)):
    #     l = np.append(l, np.zeros((len(a1) - len(l),), dtype=int))
    # if(len(u) < len(a1)):
    #     u = np.append(u, np.full((len(a1) - len(u),), 100, dtype=int))
    # if(len(d) < len(a1)):
    #     d = np.append(d, np.zeros((len(a1) - len(d),), dtype=int))
    pass
    # a1 = 0.25 * (r + l + u + d)
    
#     # a1[1:-1,1:-1] = .25 * (a0[2:,1:-1] + a0[:-2,1:-1] + a0[1:-1,2:] + a0[1:-1,:-2])
@jacobi.variant(gpu)
def jacobi_gpu(j, part_size, block_size, a0, a1):
    """
    GPU kernel call to perform a single step in the Jacobi iteration.
    """
    threads_per_block_x = 32
    # threads_per_block_y = 1024//threads_per_block_x
    # print(a1.shape[0])
    blocks_per_grid_x = (a1.shape[0] + (threads_per_block_x - 1)) // threads_per_block_x
    # blocks_per_grid_y = (a1.shape[0] + (threads_per_block_y - 1)) // threads_per_block_y

    nb_stream = stream_cupy_to_numba(cp.cuda.get_current_stream())
    gpu_jacobi_kernel[(blocks_per_grid_x), (threads_per_block_x), nb_stream](j, part_size, block_size, a0, a1)
    nb_stream.synchronize()


@numba.cuda.jit
def gpu_jacobi_kernel(j, part_size, block_size, a0, a1):
    """
    Actual CUDA kernel to do a single step.
    """
    # pass
    i = numba.cuda.grid(1)
    # compute right boundary indices
    # print(i)
    # # print(a1[i])
    if 0 < i < a1.shape[0]-1:
        r_idx = -1
        if(j % 2 != 0):
            if(i % part_size == part_size - 1):
                r_idx = -1
            else:
                r_idx = i + 1
        else:
            if(i % part_size == part_size - 1):
                r_idx = block_size + (i // part_size)
            else:
                r_idx = i + 1
        # compute left boundary indices
        l_idx = -1
        if(j % 2 == 0):
            if(i % part_size == 0):
                l_idx = -1
            else:
                l_idx = i - 1
        else:
            if(i % part_size == 0):
                l_idx = block_size + (i // part_size)
            else:
                l_idx = i - 1
        
        # print(part_size, a1.shape[0], a0.shape[0], j, i, r_idx)
        # print(j, i, l_idx)
        # print(j, i, u_idx)
        # print(j, i, d_idx)
        # s = a0.shape[0]
        # print(s[0])
        # print(type(a0))
        # print(a1.shape[0])
        # assert r_idx < s
        # assert l_idx < s
        # assert u_idx < s
        # assert d_idx < s
        # assert r_idx >= -1
        # assert l_idx >= -1
        # assert u_idx >= -1
        # assert d_idx >= -1
        # assert i < s
        # print(f'{j} {i} {r_idx}')
        # print(f'{j} {i} {l_idx}')
        # print(f'{j} {i} {u_idx}')
        # print(f'{j} {i} {d_idx}')
        r_val = 0 if r_idx == -1 else a0[r_idx]
        l_val = 0 if l_idx == -1 else a0[l_idx]
        u_val = 100 
        d_val = 0
        a0[i] = 0.25 * (r_val + l_val + u_val + d_val)


def main():


    #divisions = len(get_all_devices())*2
    devs = gpu.devices
    
    divisions = ngpus
    # print(divisions)
    
    # Set up an "n" x "n" grid of values and run
    # "steps" iterations of the 4 point stencil on it.
    n = args.n
    matrix_size = n *n
    steps = 1
    blocks = ngpus
    # block_size = n // ngpus
    part_size = n // ngpus
    block_size = matrix_size // ngpus
    # part_width = 2 * n // ngpus
    # part_length = 

    @spawn(placement=cpu)
    async def main_jacobi():
        # Set up two arrays containing the input data.
        # This demo uses the standard technique of computing
        # from one array into another then swapping the
        # input and output arrays for the next iteration.
        # These are the two arrays that will be swapped back
        # and forth as input and output.
        
        a0 = np.random.rand(matrix_size).astype('f')  
        a1 = a0.copy()

        # An object that distributes arrays across all the given devices.
        num_part = blocks*part_size
        partition = []
        itr = 0
        for i in range(0, matrix_size, part_size):
            loc = xp.gpu(itr % 2)
            partition.append(loc)
            itr += 1
        # print(partition)
        a0_crosspy = xp.array(a0, distribution=partition, axis=0)
        a1_crosspy = xp.array(a1, distribution=partition, axis=0)
        #print(a0_crosspy)
        
        device_to_array_map = {}
        itr = divisions*part_size
        for slice, device_id in a0_crosspy.device_map.items():
            if(device_id.id in device_to_array_map):
                temp = device_to_array_map.get(device_id.id)
            else:
                temp = []
            slice_new = slice[0]
            for i in range(slice_new[0], slice_new[1]):
                temp.append(i)
            # temp.append(slice_new[1] - 1)
            device_to_array_map[device_id.id] = temp
        #print(device_to_array_map)
        # print(a0_crosspy)
        # print(a0_crosspy.devices)

        for k in range(num_tests):

            # Specify which set of blocks is used as input or output
            # (they will be swapped for each iteration).

            in_blocks = a0_crosspy
            out_blocks = a1_crosspy
            # print(in_blocks.devices)
            # print(out_blocks.devices)
            # print(len(in_blocks.devices))
            # print(len(out_blocks.devices))


            # Create a set of labels for the tasks that perform the first
            # Jacobi iteration step.
            tslist = list()
            # print("added tlist")
            previous_block_tasks = []
            # print("added completed")

            start = time.perf_counter()

            # Now create the tasks for subsequent iteration steps.
            for i in range(steps):

                # Swap input and output blocks for the next step.
                # print("In_blocks len: {}".format(len(in_blocks)))
                # print("Out blocks len: {}".format(len(out_blocks)))
                in_blocks, out_blocks = out_blocks, in_blocks
                # Create a new set of labels for the tasks that do this iteration step.

                current_block_tasks = TaskSpace("block_tasks[{}]".format(i))
                # print("added current")
                # Create the tasks to do the i'th iteration.
                # As before, each task needs the following info:
                #  a block index "j"
                #  a "device" where it should execute (supplied by mapper used for partitioning)
                #  the "in_block" of data used as input
                #  the "out_block" to write the output to
                

                for j in range(divisions):

                    # if args.fixed:
                    #     device = mapper.device(j)
                    # else:
                    #     device = devs
                    device = gpu(j)
                    right_boundary = []
                    left_boundary = []
                    up_boundary = []
                    down_boundary = []
                    for i in range(n):
                        if(j % 2 == 0):
                            right_boundary.append(j * part_size * n + part_size + i * n)
                        if(j % 2 != 0):
                            left_boundary.append(i * n + part_size - 1)
                            #left_boundary.append(j * part_size * n + part_size + i * (n - 1))
                    st = time.perf_counter()
                    #print(right_boundary)
                    #print(left_boundary)
                    #print(up_boundary)
                    #print(down_boundary)
                    # Make each task operating on each block depend on the tasks for
                    # that block and its immediate neighbors from the previous iteration.
                    # m_rng = nvtx.start_range(message="task", color="red")
                    # print(previous_block_tasks[max(0, j-1):min(divisions, j+2)])
                    # print("added boundaries")
                    @spawn(current_block_tasks[j],
                           dependencies=[previous_block_tasks[max(0, j-1):min(divisions, j+2)]],
                           placement=device)
                    def device_local_jacobi_task():
                        # print("+Jacobi Task", i, j, flush=True)
                        # Read boundary values from adjacent blocks in the partition.
                        # This may communicate across device boundaries.

                        stream = cp.cuda.get_current_stream()
                        # data_start = time.perf_counter()
                        
                        # data_end = time.perf_counter()
                        #stream.synchronize()
                        #print("Data: ", data_end - data_start, flush=True)

                        # Run the computation, dispatching to device specific code.


                        #print(in_block.shape, out_block.shape, flush=True)
                        interior = device_to_array_map[j]
                        print(f'interior len: {len(interior)}')

                        s1 = time.perf_counter()
                        # rng = nvtx.start_range(message="copy", color="blue")
                        # print(type(in_blocks))
                        # print(type(interior))
                        # print(type(right_boundary))
                        # print(type(left_boundary))
                        # print(type(up_boundary))
                        # print(type(down_boundary))
                        
                        in_block = in_blocks[interior + right_boundary + down_boundary + left_boundary + up_boundary].to(device)
                        # print("In block len: {} {}".format(j, len(in_block)))
                        out_block = out_blocks[interior].to(device)
                        stream.synchronize()
                        e1 = time.perf_counter()
                        t1 = e1 - s1
                        # nvtx.end_range(rng)
                        print("Done copying: {}".format(t1), flush=True)
                        
                        compute_start = time.perf_counter()
                        #rng2 = nvtx.start_range(message="jacobi", color="green")
                        jacobi(j, part_size, block_size, in_block, out_block)
                        # out_block = out_block1
                        stream.synchronize()
                        # nvtx.end_range(rng2)
                        compute_end = time.perf_counter()
                        # print("Compute: ", compute_end - compute_start, flush=True)

                # For the next iteration, use the newly created tasks as
                # the tasks from the previous step.

                tslist.append(previous_block_tasks)
                previous_block_tasks = current_block_tasks

            await current_block_tasks
            end = time.perf_counter()
            # nvtx.end_range(m_rng)
            print("Time: ", end - start)

            # This depends on all the tasks from the last iteration step.
            #for j in range(divisions):
            #    start_index = 1 if j > 0 else 0
            #    end_index = -1 if j < divisions - 1 else None  # None indicates the last element of the dimension
            #    copy(a1[mapper.slice(j, len(a1))], out_blocks[j][start_index:end_index])
        
        del a0
        del a1

if __name__ == '__main__':
    with Parla():
        main()
