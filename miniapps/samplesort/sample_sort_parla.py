import numpy as np
import cupy as cp
from parla import Parla
from parla.tasks import spawn, TaskSpace, specialize
from parla.devices import cpu, gpu
import time
import argparse
import crosspy as xp
from profile import profile_t
import enum

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

def partitioned_crosspy(global_size: int, num_gpu:int):
    x_list = list()
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            l_sz = (((i+1) * global_size) // num_gpu -  (i * global_size) // num_gpu)
            x_list.append(cp.random.uniform(0, 1, size=l_sz, dtype = cp.float64))

    return xp.array(x_list, axis=0)

def alltoallv(sbuff : xp.array , scounts : np.array):
    num_gpus = sbuff.ndevices
    
    rcounts  = np.transpose(scounts)

    soffsets = np.zeros_like(scounts)
    roffsets = np.zeros_like(rcounts)

    for i in range(num_gpus):
        soffsets[i,:] = np.cumsum(np.append(np.array([0],dtype=np.int64), scounts[i,:]))[:-1]
        roffsets[i,:] = np.cumsum(np.append(np.array([0],dtype=np.int64), rcounts[i,:]))[:-1]
    
    recieve_partitions = [roffsets[i,-1] + rcounts[i,-1] for i in range(num_gpus)]
    
    a=list()
    for i in range(num_gpus):
        with cp.cuda.Device(i):
            a.append(cp.zeros(recieve_partitions[i]))

    rbuff = xp.array(a, axis=0)

    for i in range(num_gpus):
        for j in range(num_gpus):
            rbuff.blockview[i][roffsets[i,j] : roffsets[i,j] + rcounts[i,j]] = cp.asarray(sbuff.blockview[j][soffsets[j, i] : soffsets[j, i] + scounts[j, i]])

    return rbuff


def sync(device_ids):
    for device_id in device_ids:
        with cp.cuda.Device(device_id):
            cp.cuda.runtime.deviceSynchronize()


def parla_sample_sort(x:xp.array):
    T[pp.ALL].start()
    num_gpu     = x.ndevices
    
    a = [None] * num_gpu
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
                    
                    return
                
            await T
    
    y = xp.array(a, axis=0)
    sync([i for i in range(num_gpu)])
    T[pp.LSORT1].stop()

    if num_gpu==1:
        return y

    T[pp.SP].start()
    sp        = np.zeros((num_gpu, num_gpu-1))
    with Parla():
        @spawn(placement=cpu, vcus=0)
        async def select_splitters():
            T = TaskSpace("T")
            for i in range(num_gpu):
                @spawn(T[i], placement=[gpu(i)], vcus=0.0)
                def t1():
                    with cp.cuda.Device(i):
                        idx     = np.array([(((j+1) * len(y.blockview[i])) // num_gpu)-1 for j in range(num_gpu-1)], dtype=np.int64)
                        assert (idx>0).all()==True
                        sp[i,:] = cp.asnumpy(y.blockview[i][idx])
            await T
    
    sp = sp.reshape((num_gpu * (num_gpu-1)))
    sp = np.sort(sp)

    num_splitters = num_gpu-1
    idx = np.array([((i) * len(sp) // num_splitters) + (((i+1) * len(sp) // num_splitters) - ((i) * len(sp) // num_splitters))//2 for i in range(num_splitters)],dtype=np.int64)
    sp  = sp[idx]

    sp_gpu = [None] * num_gpu
    with Parla():
        @spawn(placement=cpu, vcus=0)
        async def bcast_splitters():
            T = TaskSpace("T")
            for i in range(num_gpu):
                @spawn(T[i], placement=[gpu(i)], vcus=0.0)
                def t1():
                    with cp.cuda.Device(i):
                        sp_gpu[i] = cp.asarray(sp)
                    return
            await T
        
        
    sp = xp.array(sp_gpu,axis=0)
    T[pp.SP].stop()

    T[pp.S_MAP].start()
    # local_counts = np.array([len(x.blockview[i]) for i in range(num_gpu)], dtype=np.int64)
    # local_offset = np.append(np.array([0], dtype=np.int64), local_counts)
    # local_offset = np.cumsum(local_offset)[:-1]

    send_count  = np.zeros((num_gpu,num_gpu), dtype=np.int64)
    #send_offset = np.zeros((num_gpu,num_gpu), dtype=np.int64)
    #gid_send    = np.zeros(y.shape[0], dtype=np.int64)

    with Parla():
        @spawn(placement=cpu, vcus=0)
        async def compute_send_counts():
            T = TaskSpace("T")
            for i in range(num_gpu):
                @spawn(T[i], placement=[gpu(i)], vcus=0.0)
                def t1():
                    with cp.cuda.Device(i):
                        idx           = cp.where(y.blockview[i]<sp.blockview[i][0])[0]
                        send_count[i,0] = len(idx)
                
                        for sp_idx in range(1, num_splitters):
                            cond = cp.logical_and(y.blockview[i] >= sp.blockview[i][sp_idx-1] ,  y.blockview[i] < sp.blockview[i][sp_idx])
                            idx  = cp.where(cond)[0]
                            send_count[i, sp_idx] = len(idx)
                    
                        idx = cp.where(y.blockview[i]>=sp.blockview[i][num_splitters-1])[0]
                        send_count[i, num_gpu-1] = len(idx)
            await T 
    
    # for i in range(num_gpu):
    #     send_offset[i,:]= np.append(np.array([0], dtype=np.int64), np.cumsum(send_count[i,:]))[:-1]

    # a=[None] * num_gpu
    # with Parla():
    #     @spawn(placement=cpu, vcus=0)
    #     async def alloc_array_sorted():
    #         T = TaskSpace("T")
    #         for i in range(num_gpu):
    #             @spawn(T[i], placement=[gpu(i)], vcus=0.0)
    #             def t1():
    #                 with cp.cuda.Device(i):
    #                     a[i] = cp.zeros(np.sum(send_count[:,i]))

    #                 return
    #         await T

    # z=xp.array(a, axis=0)
    # recieve_counts  = np.array([len(z.blockview[i]) for i in range(num_gpu)], dtype=np.int64)
    # recieve_offset  = np.append(np.array([0]), recieve_counts)
    # recieve_offset  = np.cumsum(recieve_offset)[:-1]

    # for i in range(num_gpu):
    #     tmp      = np.array([],dtype=np.int64)
    #     for j in range(num_gpu):
    #         tmp=np.append(tmp, local_offset[j] + send_offset[j,i] + np.array(range(send_count[j,i]), dtype=np.int64))
        
    #     gid_send[recieve_offset[i] : recieve_offset[i] + recieve_counts[i]] = tmp
    T[pp.S_MAP].stop()

    T[pp.ALL2ALL].start()
    # gid_recv   = np.array(range(y.shape[0]), dtype=np.int64)
    # assignment = xp.alltoall(z, gid_recv, y, gid_send)
    # assignment()
    z=alltoallv(y, send_count)
    sync([i for i in range(num_gpu)])
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
                        a[i] = cp.sort(z.blockview[i])
                    return
                
            await T
    
    z = xp.array(a, axis=0)
    sync([i for i in range(num_gpu)])
    T[pp.LSORT2].stop()
    T[pp.ALL].stop()
    return z


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n"   , "--n"       , help="global size", type=int, default=40)
    parser.add_argument("-gpu" , "--gpu"     , help="gpus"       , type=int, default= 4)
    parser.add_argument("-w"   , "--warm_up" , help="# warmup"   , type=int, default= 2)
    parser.add_argument("-r"   , "--runs"    , help="# runs"     , type=int, default= 5)

    args   = parser.parse_args()

    for iter in range(args.warm_up):
        x         = partitioned_crosspy(args.n, args.gpu)
        y         = parla_sample_sort(x)

        y_cpu = xp.asnumpy(y) 
        x_cpu = xp.asnumpy(x) 
        
        x_cpu=np.sort(x_cpu)
        print("sample sort passed : ",((y_cpu==x_cpu).all()==True))
        
    for iter in range(args.runs):
        x         = partitioned_crosspy(args.n, args.gpu)
        y         = parla_sample_sort(x)
        
        y_cpu = xp.asnumpy(y)
        x_cpu = xp.asnumpy(x)
        
        print("sample sort passed : ",((y_cpu==np.sort(x_cpu)).all()==True))
        
    
    
    profile_summary_dump(T)
    
    

    

    

    
    
    


