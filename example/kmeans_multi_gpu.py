import numpy as np
import cupy as cp

import random
import math
import time
from functools import reduce

from parla import parray
from parla import Parla, spawn, TaskSpace
from parla.cython.device_manager import cpu, gpu
from parla.common.globals import get_current_context

import nvtx

#import pdb

async def fit_xp(X, centers, sum_list,  count_list, gpu_sum, gpu_count, num_tasks, n_clusters, max_iter, num_gpu):
    #Tk = TaskSpace('c')
    #@spawn(Tk[0], inout=[(X[j],0), (centers,0), (sum_list[j],0), (count_list[j],0)], placement = [cpu(0)])
    #async def spawner():
    #assert X.ndim == 2

    Tk = TaskSpace("T")
    Tr = TaskSpace("reduction")
    Tc = TaskSpace("calc_centroids")

    nvtx.push_range(message="USER LAUNCH TASKS", color="green",
                    domain="application")

    for i in range(max_iter):
        for j in range(num_gpu):
            for k in range(num_tasks):
                @spawn(Tk[i,j,k], Tc[i-1,j] ,input=[(X[j*num_tasks+k],0)], inout=[(sum_list[j][k],0), (count_list[j][k],0), (centers[j],0)], placement = [gpu(j)])
                def kmeans_iter():

                    nvtx.push_range(message="USER TASKS 1", color="blue",
                                    domain="application")

                    samples = X[j*num_tasks+k]          
                    sums = sum_list[j][k]
                    counts = count_list[j][k]
                    cent_dev = centers[j]
                
                    # Compute the new label for each sample.
                    distances = cp.linalg.norm(samples.array[:, None, :] - cent_dev.array[None, :, :], axis=2)  # Q: Does this work? can parrays be passed to cupy functions?
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

                    nvtx.pop_range(domain="application")

            # await Tk[i,j,0:num_tasks]
            # print("Done Tk")

            # Do reduction from multiple tasks
            @spawn(Tr[i,j], [Tk[i,j,0:num_tasks]], inout=[(gpu_sum[j],0), (gpu_count[j],0)] , input=[(sum_list[j][x],0) for x in range(num_tasks)].extend([(count_list[j][x],0) for x in range(num_tasks)]), placement = [gpu(j)])
            def reduc():
                gpu_sum[j].array[:,:] = cp.stack([sum_list[j][x].array for x in range(num_tasks)]).sum(axis=0)
                gpu_count[j].array[:] = cp.stack([count_list[j][x].array for x in range(num_tasks)]).sum(axis=0)
                stream = cp.cuda.get_current_stream()
                stream.synchronize()
                #print(gpu_sum[j])
        
        # await Tr[i,0:num_gpu]
        # print("Done Tr")
        
        for j in range(num_gpu):
            @spawn(Tc[i,j], [Tr[i,0:num_gpu]], inout=[(centers[j],0)], input=[(gpu_sum[x],0) for x in range(num_gpu)]+[(gpu_count[x],0) for x in range(num_gpu)], placement = [gpu(j)]) #
            def calc():
                nvtx.push_range(message="USER TASKS 2", color="blue",
                                domain="application")
                
                num = cp.stack([gpu_sum[x].array for x in range(num_gpu)]).sum(axis=0)
                den = cp.stack([gpu_count[x].array for x in range(num_gpu)]).sum(axis=0)
                centers[j].array[:,:] = num / den 
                stream = cp.cuda.get_current_stream()
                stream.synchronize()

                nvtx.pop_range(domain="application")
        
        await Tc[i,0:num_gpu]

    nvtx.pop_range(domain="application")
    await Tc[max_iter-1,0:num_gpu]


def main():
    @spawn(placement=cpu)
    async def kmeans():
        # define Constants
        num = 5000000 #5000000
        n_clusters = 10
        num_tasks = 4
        iter = 20
        dim = 10
        num_gpus = 4 #cp.cuda.runtime.getDeviceCount()
        

        print("Generating Data points")
        samples = np.random.randn(num, dim)
        X_train = np.r_[samples + 1, samples - 1]

        # Generate initial centroids
        n_samples = len(X_train)
        initial_indexes = np.random.choice(n_samples, n_clusters, replace=False)
        
        assert n_samples%(num_tasks*num_gpus) == 0

        X_list = list()
        sum_list_cp = list()
        count_list_cp = list()
        centers_cp = list()
        gpu_sum_cp = list()
        gpu_count_cp = list()
        block_size = int(n_samples/(num_tasks*num_gpus))
        for j in range(num_gpus):
            sum_list_cp.append(list())
            count_list_cp.append(list())
            with cp.cuda.Device(j) as dev:
                centers_cp.append(cp.asarray(X_train[initial_indexes]))
                gpu_sum_cp.append(cp.zeros((n_clusters, dim)))
                gpu_count_cp.append(cp.zeros((n_clusters,1)))
                for i in range(num_tasks): 
                    arr = cp.asarray(X_train[(num_tasks*j+i)*block_size:(num_tasks*j+i+1)*block_size], order='F')
                    X_list.append(arr)
                    sum_list_cp[j].append(cp.zeros((n_clusters, dim)))
                    count_list_cp[j].append(cp.zeros((n_clusters,1)))
                    cp.cuda.Device().synchronize()

        print("Generated Data")
        # Make PArrays
        A = parray.asarray_batch(X_list)        # Q:Is this okay?
        centers = parray.asarray_batch(centers_cp)
        gpu_sum = parray.asarray_batch(gpu_sum_cp)
        gpu_count = parray.asarray_batch(gpu_count_cp)
        sum_list = list()
        count_list = list()
        for i in range(num_gpus):
            sum_list.append(parray.asarray_batch(sum_list_cp[i])) #, on_gpu=True
            count_list.append(parray.asarray_batch(count_list_cp[i]))

        print("Starting kmeans")
        print("------------")
        start = time.perf_counter()
        # calculate kmeans for iter iterations
        await fit_xp(A, centers, sum_list, count_list, gpu_sum, gpu_count, num_tasks, n_clusters, iter, num_gpus)
        
        end = time.perf_counter()
        print("Time = ", end-start)
        # print(type(centers))
        # print(centers.array.shape)
        # print(type(sum_list))
        # print(type(sum_list[0]))
        # print(type(count_list))
        # print(count_list.array.shape)

        print("Copy Back")
        ts = TaskSpace("CopyBack")
        @spawn(ts[0], input=[(centers[0], 0)] , placement=cpu)
        def copy_back():
            centers_cpu = centers[0].array
            print(centers_cpu)
            
        await ts[0]
        
        #test results
        test = 0
        if test == 1:    
            centers_test = X_train[initial_indexes]
            for _ in range(iter):
                sum = np.zeros((n_clusters,dim))
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