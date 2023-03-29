import pytest

from parla import Parla, spawn, TaskSpace, parray
from parla.common.parray.coherence import Coherence
from parla.common.globals import get_scheduler, get_device_manager
from parla.cython.device_manager import cpu, cuda
import numpy as np

def test_parray_task():
    with Parla():
        @spawn(placement=cpu)
        async def main():
            n = 2
            np.random.seed(10)
            # Construct input data
            a = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            b = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            a = parray.asarray(a)
            b = parray.asarray(a)
            ts = TaskSpace("Tracking")
            scheduler = get_scheduler()
            device_manager = get_device_manager()
            num_devices = len(device_manager.get_all_devices())

            ## Simple test for active task of PArray.

            for i in range(0, num_devices):
                assert a.get_num_active_tasks(i) == 0

            @spawn(ts[1], placement=[cuda(1)], input=[(a, 0)])
            def task1():
                assert a.get_num_active_tasks(2) == 1
                
            await ts
            @spawn(ts[2], placement=[cuda(3)], inout=[(a, 0)])
            def task2():
                assert a.get_num_active_tasks(4) == 1 or a.get_num_active_tasks(4) == 2
                assert a.get_num_active_tasks(2) == 0

            @spawn(ts[3], placement=[cuda(3)], inout=[(a, 0)])
            def task3():
                assert a.get_num_active_tasks(4) == 1 or a.get_num_active_tasks(4) == 2
            await ts

            ## Simple test for active task of PArray slicing.


            ## Simple test for PArray tracker memory management.


if __name__=="__main__":
    test_parray_task()
