import numpy as np
import cupy as cp

import random
import math

from parla import parray
from parla import Parla, spawn, TaskSpace
from parla.cython.device_manager import cpu, gpu
from parla.common.globals import get_current_context

import pdb

async def fit_xp(X, centers, sum_list,  count_list,  num_tasks, n_clusters, max_iter):
    Tk = TaskSpace('c')
    @spawn(Tk[0], inout=[(X[j],0), (centers,0), (sum_list[j],0), (count_list[j],0)], placement = [cpu(0)])
    async def spawner():
        #assert X.ndim == 2
        T = TaskSpace("T")
        for i in range(max_iter):
            for j in range(num_tasks):

                @spawn(T[i,j], input=[(X[j],0), (centers,0)], output=[(sum_list[j],0), (count_list[j],0)], placement = [gpu(0)])
                def kmeans_iter():
                    samples = X[j]          
                    #sums = sum_list[j]
                    #counts = count_list[j]
                    # Compute the new label for each sample.
                    distances = cp.linalg.norm(samples.array[:, None, :] - centers.array[None, :, :], axis=2)  # Q: Does this work? can parrays be passed to cupy functions?
                    pred = cp.argmin(distances, axis=1) #rename new_

                    # If the label is not changed for each sample, we suppose the
                    # algorithm has converged and exit from the loop.
                    #if xp.all(new_pred == pred):
                    #    break
                    #pred = new_pred

                    # Compute the new centroid for each cluster.
                    i = cp.arange(n_clusters)
                    mask = pred == i[:, None]
                    sum_list[j].array[:,:] = cp.where(mask[:, :, None], samples.array, 0).sum(axis=1)
                    count_list[j].array[:] = cp.count_nonzero(mask, axis=1).reshape((n_clusters, 1))
                    stream = cp.cuda.get_current_stream()
                    stream.synchronize()
                
            await T[i,:]

            Ts = TaskSpace("reduction")
            @spawn(Ts[0], output=[(centers,0)], input=[(sum_list,0), (count_list,0)], placement = [gpu(0)])
            def reduc():
                centers[:,:] = sum_list.sum(axis=0) / count_list.sum(axis=0)
                stream = cp.cuda.get_current_stream()
                stream.synchronize()

            await Ts[0]

    await Tk[0]
    #return centers


async def main():
    #@spawn(placement=cpu)
    #async def kmeans():
    num = 100000
    n_clusters = 10
    num_tasks = 10
    
    samples = np.random.randn(num, 2)
    X_train = np.r_[samples + 1, samples - 1]

    n_samples = len(X_train)
    initial_indexes = np.random.choice(n_samples, n_clusters, replace=False)
    centers_cpu = X_train[initial_indexes]

    breakpoint()

    X_list = []
    sum_list_cp = []
    count_list_cp = []
    block_size = int(num / num_tasks)
    for i in range(num_tasks):
        with cp.cuda.Device(0) as dev:
            arr = cp.asarray(X_train[i*block_size:(i+1)*block_size], order='F')
            X_list.append(arr)
            sum_list_cp.append(cp.zeros((n_clusters, 2)))
            count_list_cp.append(cp.zeros((n_clusters,1)))
            cp.cuda.Device().synchronize()

    # Convert X_list (shape = (num_tasks,block_size,2)) and others to parrays 
    A = parray.asarray_batch(X_list)        # Q:Is this okay?
    centers = parray.asarray(centers_cpu, on_gpu=True)
    sum_list = parray.asarray(sum_list_cp, on_gpu=True)
    count_list = parray.asarray(count_list_cp, on_gpu=True)

    await fit_xp(A, centers, sum_list, count_list, num_tasks, n_clusters, 1)

    print("Copy Back")
    #for i in range(num_tasks):
    #    X_train[i*block_size:(i+1)*block_size] = A[i].array
    #centers_cpu = cp.asnumpy(centers.array)     # Q: is this ok?
    #print(type(centers))

if __name__ == "__main__":
    with Parla():
        main()