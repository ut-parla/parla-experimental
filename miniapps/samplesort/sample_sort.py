import numpy as np
import cupy as cp
from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu, gpu
import time
import argparse
import crosspy as xp
import asyncio
from time import perf_counter as time

class profile_t:
    def __init__(self,name):
        self.name = name
        self.seconds=0
        self.snap=0
        self._pri_time =0
        self.iter =0

    def __add__(self,o):
        assert(self.name==o.name)
        self.seconds+=o.seconds
        self.snap+=o.snap
        self.iter+=o.iter
        return self

    def start(self):
        self._pri_time = time()
    
    def stop(self):
        self.seconds-=self._pri_time
        self.snap=-self._pri_time

        self._pri_time = time()

        self.seconds +=self._pri_time
        self.snap  += self._pri_time
        self.iter+=1
    
    def reset(self):
        self.seconds=0
        self.snap=0
        self._pri_time =0
        self.iter =0


def partitioned_crosspy(global_size: int, num_gpu:int):

    x_list = list()
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            l_sz = (((i+1) * global_size) // num_gpu -  (i * global_size) // num_gpu)
            x_list.append(cp.random.uniform(0, 1, size=l_sz, dtype = cp.float64))

    return xp.array(x_list, axis=0)

def sync(ngpus):
    for i in range(ngpus):
        with cp.cuda.Device(i):
            cp.cuda.runtime.deviceSynchronize()

def crosspy_sample_sort(x:xp.array):
    t_all       = profile_t("all")

    t_all.start()
    num_gpu     = x.ndevices
    
    t_local_sort1 = profile_t("local_sort1")
    t_local_sort1.start()
    u         = xp.array(list(x.block_view(cp.sort)), axis=0)
    sync(4)
    print(u.device_map)
    t_local_sort1.stop()
    
    if num_gpu == 1:
        t_all.stop()
        profile_ctx = [t_all, t_local_sort1]
        return u, profile_ctx

    t_splitters = profile_t("splitters")
    t_splitters.start()
    sp        = np.zeros(num_gpu * (num_gpu-1))
    u_blkview = u.blockview
    
    for i in range(num_gpu):
        y   = u_blkview[i]
        idx = np.array([(((j+1) * len(y)) // num_gpu)-1 for j in range(num_gpu-1)], dtype=np.int64)
        assert (idx>0).all()
        with cp.cuda.Device(i):
            sp[i * (num_gpu-1) : (i+1) * (num_gpu-1)] = cp.asnumpy(y[idx])

    sync(4)
    sp  = np.sort(sp)
    num_splitters = num_gpu-1

    idx = np.array([((i) * len(sp) // num_splitters) + (((i+1) * len(sp) // num_splitters) - ((i) * len(sp) // num_splitters))//2 for i in range(num_splitters)],dtype=np.int64)
    sp  = sp[idx]
    t_splitters.stop()


    t_splitter_bcast = profile_t("splitters_bcast")
    t_splitter_bcast.start()
    sp_list = []
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            sp_list.append(cp.asarray(sp))
    
    sync(4)
    sp = xp.array(sp_list, axis=0)
    sp_blkview = sp.blockview
    sync(4)
    t_splitter_bcast.stop()


    t_splitters_smap = profile_t("splitters_smap")
    t_splitters_smap.start()
    # to compute global ids
    local_counts = np.array([len(u_blkview[i]) for i in range(num_gpu)])
    local_offset = np.append(np.array([0]), local_counts)
    local_offset = np.cumsum(local_offset)[:-1]

    send_count  = np.zeros((num_gpu,num_gpu), dtype=np.int64)
    send_offset = np.zeros((num_gpu,num_gpu), dtype=np.int64)
    gid_send = np.zeros(u.shape[0], dtype=np.int64)

    for i in range(num_gpu):
        with cp.cuda.Device(i):
            idx           = cp.where(u_blkview[i]<sp_blkview[i][0])[0]
            send_count[i,0] = len(idx)
            
            for sp_idx in range(1, num_splitters):
                cond = cp.logical_and(u_blkview[i]>=sp_blkview[i][sp_idx-1],  u_blkview[i]<sp_blkview[i][sp_idx])
                idx  = cp.where(cond)[0]
                send_count[i, sp_idx] = len(idx)
        
            idx = cp.where(u_blkview[i]>=sp_blkview[i][num_splitters-1])[0]
            send_count[i, num_gpu-1] = len(idx)
    
    sync(4)
    
    for i in range(num_gpu):
        send_offset[i,:]= np.append(np.array([0], dtype=np.int64), np.cumsum(send_count[i,:]))[:-1]
    
    arr_list = list()
    for i in range(num_gpu):
        with cp.cuda.Device(i):
            arr_list.append(cp.zeros(np.sum(send_count[:,i])))
    
    sync(4)
    rbuff         = xp.array(arr_list, axis=0)
    rbuff_blkview = rbuff.blockview

    rbuff_counts  = np.array([len(rbuff_blkview[i]) for i in range(num_gpu)])
    rbuff_offset  = np.append(np.array([0]), rbuff_counts)
    rbuff_offset  = np.cumsum(rbuff_offset)[:-1]

    for i in range(num_gpu):
        tmp      = np.array([],dtype=np.int64)
        for j in range(num_gpu):
            tmp=np.append(tmp, local_offset[j] + send_offset[j,i] + np.array(range(send_count[j,i]), dtype=np.int64))
            
        gid_send[rbuff_offset[i] : rbuff_offset[i] + rbuff_counts[i]] = tmp

    t_splitters_smap.stop()

    t_all2all_node = profile_t("all2allv_node")
    t_all2all_node.start()
    gid_recv   = np.array(range(u.shape[0]), dtype=np.int64)
    assignment = xp.alltoall(rbuff, gid_recv, u, gid_send)
    sync(4)
    assignment()
    sync(4)
    
    t_all2all_node.stop()

    t_local_sort2 = profile_t("local_sort2")
    t_local_sort2.start()
    rbuff  = xp.array(list(rbuff.block_view(cp.sort)), axis=0)
    sync(4)
    t_local_sort2.stop()

    profile_ctx = [t_all, t_local_sort1, t_splitters, t_splitter_bcast, t_splitters_smap, t_all2all_node, t_local_sort2]
    t_all.stop()
    return rbuff, profile_ctx

def profile_summary_dump(t_ctx_list):
    runs  = len(t_ctx_list)

    # for i in range(0, runs):
    #     for j, t in enumerate(t_ctx_list[i]):
    #         print(i, "%.4E"%t.seconds)

    for i in range(1, runs):
        for j, t in enumerate(t_ctx_list[i]):
            t_ctx_list[0][j] +=(t)
    
    
    for i, t in enumerate(t_ctx_list[0]):
        if i==0:
            print("%s -- %.4E"%(t.name, t.seconds/runs))
        else:
            print("  |%s -- %.4E"%(t.name, t.seconds/runs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n"   , "--n"       , help="global size", type=int, default=40)
    parser.add_argument("-gpu" , "--gpu"     , help="gpus"       , type=int, default= 4)
    parser.add_argument("-w"   , "--warm_up" , help="# warmup"   , type=int, default= 2)
    parser.add_argument("-r"   , "--runs"    , help="# runs"     , type=int, default= 5)

    args   = parser.parse_args()

    for iter in range(args.warm_up):
        x         = partitioned_crosspy(args.n, args.gpu)
        y , t_ctx = crosspy_sample_sort(x)

        y_cpu = xp.asnumpy(y) #y.all_to(xp.cpu(0))
        x_cpu = xp.asnumpy(x) #x.all_to(xp.cpu(0))
        
        for i in range(args.gpu):
            with cp.cuda.Device(i):
                cp.cuda.runtime.deviceSynchronize()
        
        x_cpu=np.sort(x_cpu)
        print("sample sort passed : ",((y_cpu==x_cpu).all()==True))
        print(y_cpu[x_cpu!=y_cpu], len(y_cpu[x_cpu!=y_cpu]) )
    
    # t_ctx_list = list()

    # for iter in range(args.runs):
    #     x         = partitioned_crosspy(args.n, args.gpu)
    #     y , t_ctx = crosspy_sample_sort(x)
    #     t_ctx_list.append(t_ctx)
    #     cp.cuda.runtime.deviceSynchronize()

    #     y_cpu = xp.asnumpy(y) #y.all_to(xp.cpu(0))
    #     x_cpu = xp.asnumpy(x) #x.all_to(xp.cpu(0))
        
    #     print("sample sort passed : ",((y_cpu==np.sort(x_cpu)).all()==True))
        
    
    
    # profile_summary_dump(t_ctx_list)
    
    

    

    

    
    
    


