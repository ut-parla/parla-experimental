import numpy as np
import cupy as cp

import random
import math
import time

from parla import parray
from parla import Parla, spawn, TaskSpace
from parla.cython.device_manager import cpu, gpu
from parla.common.globals import get_current_context

#import pdb

async def fit_xp(X, centers, sum_list,  count_list,  num_tasks, n_clusters, max_iter):
    #Tk = TaskSpace('c')
    #@spawn(Tk[0], inout=[(X[j],0), (centers,0), (sum_list[j],0), (count_list[j],0)], placement = [cpu(0)])
    #async def spawner():
    #assert X.ndim == 2

    Tk = TaskSpace("T")
    Tr = TaskSpace("reduction")
    for i in range(max_iter):
        for j in range(num_tasks):

            @spawn(Tk[i,j], Tr[i-1] ,input=[(X[j],0), (centers,0)], inout=[(sum_list[j],0), (count_list[j],0)], placement = [gpu(0)])
            def kmeans_iter():
                samples = X[j]          
                sums = sum_list[j]
                counts = count_list[j]

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
                sums.array[:,:] = cp.where(mask[:, :, None], samples.array, 0).sum(axis=1)
                counts.array[:] = cp.count_nonzero(mask, axis=1).reshape((n_clusters, 1))
                stream = cp.cuda.get_current_stream()
                stream.synchronize()

        #time.sleep(5)    
        #await Tk[i,0:num_tasks]

        # Do reduction from multiple tasks
        @spawn(Tr[i], [Tk[i,0:num_tasks]], inout=[(centers,0)], input=[(sum_list,0), (count_list,0)], placement = [gpu(0)])
        def reduc():
            centers[:,:] = sum_list.array.sum(axis=0) / count_list.array.sum(axis=0)
            stream = cp.cuda.get_current_stream()
            stream.synchronize()

        # time.sleep(5)
    
    await Tr[0:max_iter]


def main():
    @spawn(placement=cpu)
    async def kmeans():
        # define Constants
        num = 100000
        n_clusters = 5
        num_tasks = 10
        iter = 2
        
        # Generate Data points
        samples = np.random.randn(num, 2)
        X_train = np.r_[samples + 1, samples - 1]

        # Generate initial centroids
        n_samples = len(X_train)
        initial_indexes = np.random.choice(n_samples, n_clusters, replace=False)
        centers_cpu = X_train[initial_indexes]

        X_list = list()
        sum_list_cp = list()
        count_list_cp = list()
        block_size = int(n_samples / num_tasks)
        for i in range(num_tasks):
            with cp.cuda.Device(0) as dev:
                arr = cp.asarray(X_train[i*block_size:(i+1)*block_size], order='F')
                X_list.append(arr)
                sum_list_cp.append(cp.zeros((n_clusters, 2)))
                count_list_cp.append(cp.zeros((n_clusters,1)))
                cp.cuda.Device().synchronize()

        # Make PArrays
        A = parray.asarray_batch(X_list)        # Q:Is this okay?
        centers = parray.asarray(centers_cpu)
        sum_list = parray.asarray(sum_list_cp, on_gpu=True)
        count_list = parray.asarray(count_list_cp, on_gpu=True)

        # calculate kmeans for iter iterations
        await fit_xp(A, centers, sum_list, count_list, num_tasks, n_clusters, iter)

        print("Copy Back")

        ts = TaskSpace("CopyBack")
        @spawn(ts[0], input=[(sum_list, 0), (count_list, 0), (centers, 0)] , placement=cpu)
        def copy_back():
            centers_cpu[:] = centers.array
            print(centers_cpu)
            
        await ts[0]
        
        #test results
        test = 1
        if test == 1:    
            centers_test = X_train[initial_indexes]
            for _ in range(iter):
                sum = np.zeros((n_clusters,2))
                count = np.zeros(n_clusters)
                for i in range(n_samples):
                    dist = np.linalg.norm(centers_test - X_train[i,:], axis=1)
                    index = np.argmin(dist, axis=0)
                    sum[index] = sum[index] + X_train[i]
                    count[index] = count[index] + 1
                
                centers_test[:] = sum / count[:,None]
            print(centers_test)


if __name__ == "__main__":
    with Parla():
        main()