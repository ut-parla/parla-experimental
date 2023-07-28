import numpy as np
import cupy as cp
from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu, gpu
import argparse
import crosspy as xp
import enum
from profile import profile_t
from threading import Thread
from time import perf_counter as time
from multiprocessing.pool import ThreadPool as WorkerPool

class pp(enum.IntEnum):
    ALL     = 0
    LSORT1  = 1
    SP      = 2
    S_MAP   = 3
    ALL2ALL = 4
    LSORT2  = 5
    LAST    = 6

T  =[None]*int(pp.LAST)
TN =["All","Local sort 1", "Splitter comp. ", "Scatter map", "All to all ", "Local Sort 2"]
for i in range(pp.LAST):
    T[i] = profile_t("T_%d"%(i))

def partitioned_crosspy(global_size: int, num_gpu:int):

    x = [None] * num_gpu
    # with Parla():
    #     @spawn(placement=cpu, vcus=0)
    #     async def local_sort():
    #         T = TaskSpace("T")
    #         for i in range(num_gpu):
    #             @spawn(T[i], placement=[gpu(i)], vcus=0.0)
    #             def t1():
    #                 with cp.cuda.Device(i):
    #                     l_sz      = (((i+1) * global_size) // num_gpu -  (i * global_size) // num_gpu)
    #                     x[i] = cp.random.uniform(0, 1, size=l_sz, dtype = cp.float64)
    #                     cp.cuda.runtime.deviceSynchronize()
    #                 return
    #         await T
    
    
    def t1(i):
        with cp.cuda.Device(i):
            l_sz = (((i+1) * global_size) // num_gpu -  (i * global_size) // num_gpu)
            x[i] = cp.random.uniform(0, 1, size=l_sz, dtype = cp.float64)
            cp.cuda.runtime.deviceSynchronize()
        return
    
    pool = WorkerPool(num_gpu)    
    pool.map(t1,[i for i in range(num_gpu)])
    pool.close()
    pool.join()
    return xp.array(x, axis=0)

def sync(device_ids):
    for device_id in device_ids:
        with cp.cuda.Device(device_id):
            cp.cuda.runtime.deviceSynchronize()

def asnumpy1(x:xp.array):
    num_gpu     = x.ndevices
    
    local_counts = np.array([len(x.blockview[i]) for i in range(num_gpu)], dtype=np.int64)
    local_offset = np.cumsum(np.append(np.array([0], dtype=np.int64), local_counts))[:-1]

    x_np         = np.zeros(np.sum(local_counts))
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            x_np[local_offset[i]:local_offset[i] + local_counts[i]] = cp.asnumpy(x.blockview[i])

    return x_np

def profile_summary_dump(T):
    for i, t in enumerate(T):
        if t.iter==0:
            continue

        if i==0:
            print("%s -- %.4E"%(TN[i], t.seconds/t.iter))
        else:
            print("  |%s -- %.4E"%(TN[i], t.seconds/t.iter))


    for i, t in enumerate(T):
        if t.iter==0:
            continue

        print("%.8E"%(t.seconds/t.iter)+",",end="")
    
    print("\n")
        
def alltoallv(sbuff : xp.array , scounts : np.array):
    num_gpu = sbuff.ndevices
    
    rcounts  = np.transpose(scounts)

    soffsets = np.zeros_like(scounts)
    roffsets = np.zeros_like(rcounts)

    for i in range(num_gpu):
        soffsets[i,:] = np.cumsum(np.append(np.array([0],dtype=np.int64), scounts[i,:]))[:-1]
        roffsets[i,:] = np.cumsum(np.append(np.array([0],dtype=np.int64), rcounts[i,:]))[:-1]

    recieve_partitions = [roffsets[i,-1] + rcounts[i,-1] for i in range(num_gpu)]
    
    a=[None] * num_gpu
    
    with Parla():
        @spawn(placement=cpu, vcus=0)
        async def _():
            T = TaskSpace("T")
            for i in range(num_gpu):
                @spawn(T[i], placement=[gpu(i)], vcus=0.0)
                def __():
                    with cp.cuda.Device(i):
                        a[i] = cp.zeros(recieve_partitions[i], dtype = sbuff.dtype)
                        for j in range(num_gpu):
                            with cp.cuda.Stream(non_blocking=True) as stream:
                                a[i][roffsets[i,j] : roffsets[i,j] + rcounts[i,j]] = cp.asarray(sbuff.blockview[j][soffsets[j, i] : soffsets[j, i] + scounts[j, i]])
                        
                        cp.cuda.runtime.deviceSynchronize()
            await T
    
    return xp.array(a, axis=0)

def alltoallv_threads(sbuff : xp.array , scounts : np.array):
    num_gpu = sbuff.ndevices
    
    rcounts  = np.transpose(scounts)

    soffsets = np.zeros_like(scounts)
    roffsets = np.zeros_like(rcounts)

    for i in range(num_gpu):
        soffsets[i,:] = np.cumsum(np.append(np.array([0],dtype=np.int64), scounts[i,:]))[:-1]
        roffsets[i,:] = np.cumsum(np.append(np.array([0],dtype=np.int64), rcounts[i,:]))[:-1]

    recieve_partitions = [roffsets[i,-1] + rcounts[i,-1] for i in range(num_gpu)]
    
    a=[None] * num_gpu
    def t1(i):
        with cp.cuda.Device(i):
            a[i] = cp.zeros(recieve_partitions[i])
            for j in range(num_gpu):
                with cp.cuda.Stream(non_blocking=True) as stream:
                    a[i][roffsets[i,j] : roffsets[i,j] + rcounts[i,j]] = cp.asarray(sbuff.blockview[j][soffsets[j, i] : soffsets[j, i] + scounts[j, i]])
            
            cp.cuda.runtime.deviceSynchronize()

    pool = WorkerPool(num_gpu)    
    pool.map(t1,[i for i in range(num_gpu)])
    pool.close()
    pool.join()

    return xp.array(a, axis=0)

def crosspy_sample_sort(x:xp.array):
    T[pp.ALL].start()
    num_gpu  = x.ndevices
    y           = [None] * num_gpu
    
    T[pp.LSORT1].start()
    def t1(i):
        with cp.cuda.Device(i):
            y[i] = cp.sort(x.blockview[i])
            cp.cuda.runtime.deviceSynchronize()          
        return
    
    pool = WorkerPool(num_gpu)    
    pool.map(t1,[i for i in range(num_gpu)])
    pool.close()
    pool.join()
    y = xp.array(y, axis=0)
    T[pp.LSORT1].stop()

    if num_gpu == 1:
        T[pp.ALL].stop()
        return y
    
    T[pp.SP].start()
    sp        = np.zeros((num_gpu, num_gpu-1))
    
    for i in range(num_gpu):
        idx = np.array([(((j+1) * len(y.blockview[i])) // num_gpu)-1 for j in range(num_gpu-1)], dtype=np.int64)
        with cp.cuda.Device(i):
            sp[i , :] = cp.asnumpy(y.blockview[i][idx])

    for i in range(num_gpu):
        with cp.cuda.Device(i):
            cp.cuda.runtime.deviceSynchronize()
    
    sp  = sp.reshape((num_gpu * (num_gpu-1)))
    sp  = np.sort(sp)
    
    num_splitters = num_gpu-1

    idx = np.array([((i) * len(sp) // num_splitters) + (((i+1) * len(sp) // num_splitters) - ((i) * len(sp) // num_splitters))//2 for i in range(num_splitters)],dtype=np.int64)
    sp  = sp[idx]
    
    a = [None] * num_gpu
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            a[i] = cp.asarray(sp)

    sp = xp.array(a, axis=0)
    T[pp.SP].stop()

    T[pp.S_MAP].start()
    send_count  = np.zeros((num_gpu,num_gpu), dtype=np.int64)
    
    def t1(i):
        with cp.cuda.Device(i):
            send_count[i,0:num_splitters]  = cp.asnumpy(cp.searchsorted(y.blockview[i], sp.blockview[i]))
            cp.cuda.runtime.deviceSynchronize()

        send_count[i,-1] = len(y.blockview[i]) - np.sum(send_count[i,-2])
        
        for j in reversed(range(1, num_gpu-1)):
            send_count[i,j] = send_count[i,j] - send_count[i,j-1]

    
    pool = WorkerPool(num_gpu)    
    pool.map(t1,[i for i in range(num_gpu)])
    pool.close()
    pool.join()
    
    T[pp.S_MAP].stop()

    T[pp.ALL2ALL].start()
    y=alltoallv_threads(y, send_count)
    T[pp.ALL2ALL].stop()

    T[pp.LSORT2].start()
    a = [None] * num_gpu
    def t1(i):
        with cp.cuda.Device(i):
            a[i] = cp.sort(y.blockview[i])
            cp.cuda.runtime.deviceSynchronize()          
        return

    pool = WorkerPool(num_gpu)    
    pool.map(t1,[i for i in range(num_gpu)])
    pool.close()
    pool.join()
    y = xp.array(a, axis=0)
    T[pp.LSORT2].stop()

    T[pp.ALL].stop()
    return y

def parla_sample_sort(x:xp.array):
    T[pp.ALL].start()
    num_gpu     = x.ndevices
    
    a           = [None] * num_gpu
    T[pp.LSORT1].start()
    with Parla():
        @spawn(placement=cpu, vcus=0)
        async def local_sort():
            T = TaskSpace("T")
            for i in range(num_gpu):
                @spawn(T[i], placement=[gpu(i)], vcus=0.0)
                def t1():
                    with cp.cuda.Device(i):
                        a[i] = cp.sort(x.blockview[i])
                        cp.cuda.runtime.deviceSynchronize()
                    return
                
            await T
    
    y = xp.array(a, axis=0)
    T[pp.LSORT1].stop()

    if num_gpu==1:
        T[pp.ALL].stop()
        return y

    T[pp.SP].start()
    sp        = np.zeros((num_gpu, num_gpu-1))
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            idx     = np.array([(((j+1) * len(y.blockview[i])) // num_gpu)-1 for j in range(num_gpu-1)], dtype=np.int64)
            sp[i,:] = cp.asnumpy(y.blockview[i][idx])

    sp = sp.reshape((num_gpu * (num_gpu-1)))
    sp = np.sort(sp)

    num_splitters = num_gpu-1
    idx = np.array([((i) * len(sp) // num_splitters) + (((i+1) * len(sp) // num_splitters) - ((i) * len(sp) // num_splitters))//2 for i in range(num_splitters)],dtype=np.int64)
    sp  = sp[idx]

    a = [None] * num_gpu
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            a[i] = cp.asarray(sp)

    for i in range(num_gpu):
        with cp.cuda.Device(i):
            cp.cuda.runtime.deviceSynchronize()

    sp = xp.array(a,axis=0)
    T[pp.SP].stop()

    T[pp.S_MAP].start()
    send_count  = np.zeros((num_gpu,num_gpu), dtype=np.int64)
    
    with Parla():
        @spawn(placement=cpu, vcus=0)
        async def local_sort():
            T = TaskSpace("T")
            for i in range(num_gpu):
                @spawn(T[i], placement=[gpu(i)], vcus=0.0)
                def t1():
                    with cp.cuda.Device(i):
                        send_count[i,0:num_splitters]  = cp.asnumpy(cp.searchsorted(y.blockview[i], sp.blockview[i]))
                        cp.cuda.runtime.deviceSynchronize()

                    send_count[i,-1] = len(y.blockview[i]) - np.sum(send_count[i,-2])
                    
                    for j in reversed(range(1, num_gpu-1)):
                        send_count[i,j] = send_count[i,j] - send_count[i,j-1]
            
            await T
    T[pp.S_MAP].stop()

    T[pp.ALL2ALL].start()
    y=alltoallv(y, send_count)
    T[pp.ALL2ALL].stop()

    T[pp.LSORT2].start()
    a = [None] * num_gpu
    with Parla():
        @spawn(placement=cpu, vcus=0)
        async def local_sort():
            T = TaskSpace("T")
            for i in range(num_gpu):
                @spawn(T[i], placement=[gpu(i)], vcus=0.0)
                def t1():
                    with cp.cuda.Device(i):
                        a[i] = cp.sort(y.blockview[i])
                        cp.cuda.runtime.deviceSynchronize()
                    return
                
            await T
    
    y = xp.array(a, axis=0)
    T[pp.LSORT2].stop()
    
    T[pp.ALL].stop()
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n"   , "--n"       , help="global size", type=int, default=40)
    parser.add_argument("-gpu" , "--gpu"     , help="gpus"       , type=int, default= 4)
    parser.add_argument("-w"   , "--warm_up" , help="# warmup"   , type=int, default= 2)
    parser.add_argument("-r"   , "--runs"    , help="# runs"     , type=int, default= 5)
    parser.add_argument("-m"   , "--mode"    , help="mode"       , type=str, default= "crosspy")
    parser.add_argument("-check" , "--check" , help="check valid sort" , type=int, default= 1)

    args   = parser.parse_args()
    print(args)
    
    if args.mode=="crosspy":
        sort_func = crosspy_sample_sort
    elif args.mode=="parla":
        sort_func = parla_sample_sort
    
    for iter in range(args.warm_up):
        x         = partitioned_crosspy(args.n, args.gpu)
        y         = sort_func(x)

        if args.check == 1 :
            y_cpu = asnumpy1(y) 
            x_cpu = asnumpy1(x) 
            x_cpu=np.sort(x_cpu)
            print("[warmup] sample sort passed : ",((y_cpu==x_cpu).all()==True))
    
    for i in range(args.gpu):
        with cp.cuda.Device(i):
            cp.cuda.runtime.deviceSynchronize()
        
    for t in T:
        t.reset()
        
    for iter in range(args.runs):
        x         = partitioned_crosspy(args.n, args.gpu)
        y         = sort_func(x)

    profile_summary_dump(T)
    
    

    

    

    
    
    


