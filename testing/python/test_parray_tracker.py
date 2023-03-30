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
                if i == 0:
                    assert scheduler.get_parray_state(i, a.parent_ID)
                else:
                    assert not scheduler.get_parray_state(i, a.parent_ID)

            @spawn(ts[1], placement=[cuda(1)], input=[(a, 0)])
            def task1():
                assert a.get_num_active_tasks(2) == 1
                for i in range(0, num_devices):
                    if i == 2 or i == 0:
                        assert scheduler.get_parray_state(i, a.parent_ID)
                    else:
                        assert not scheduler.get_parray_state(i, a.parent_ID)
            await ts

            @spawn(ts[2], placement=[cuda(3)], inout=[(a, 0)])
            def task2():
                assert a.get_num_active_tasks(4) == 1 or a.get_num_active_tasks(4) == 2
                assert a.get_num_active_tasks(2) == 0
                assert not scheduler.get_parray_state(1, a.parent_ID)
                assert scheduler.get_parray_state(4, a.parent_ID)

            @spawn(ts[3], placement=[cuda(3)], inout=[(a, 0)])
            def task3():
                assert a.get_num_active_tasks(4) == 1 or a.get_num_active_tasks(4) == 2
                assert not scheduler.get_parray_state(1, a.parent_ID)
                assert scheduler.get_parray_state(4, a.parent_ID)
            await ts

            @spawn(ts[4], placement=[cuda(3)], inout=[(a, 0)])
            def task4():
                assert a.get_num_active_tasks(1) == 0
                assert a.get_num_active_tasks(2) == 0
                assert a.get_num_active_tasks(4) == 1 or a.get_num_active_tasks(4) == 2
                assert not scheduler.get_parray_state(0, a.parent_ID)
                assert not scheduler.get_parray_state(1, a.parent_ID)
                assert not scheduler.get_parray_state(2, a.parent_ID)
                assert scheduler.get_parray_state(4, a.parent_ID)

            @spawn(ts[5], dependencies=[ts[4]], placement=[cuda(3)], input=[(a, 0)])
            def task5():
                assert a.get_num_active_tasks(1) == 0
                assert a.get_num_active_tasks(2) == 0
                assert a.get_num_active_tasks(3) == 0 or a.get_num_active_tasks(3) == 1
                assert a.get_num_active_tasks(4) == 1
                assert not scheduler.get_parray_state(0, a.parent_ID)
                assert not scheduler.get_parray_state(1, a.parent_ID)
                assert not scheduler.get_parray_state(2, a.parent_ID)
                assert scheduler.get_parray_state(4, a.parent_ID)

            @spawn(ts[6], dependencies=[ts[5]], placement=[cuda(2)], inout=[(a, 0)])
            def task6():
                assert a.get_num_active_tasks(1) == 0
                assert a.get_num_active_tasks(2) == 0
                assert a.get_num_active_tasks(3) == 1
                assert a.get_num_active_tasks(4) == 0
                assert not scheduler.get_parray_state(0, a.parent_ID)
                assert not scheduler.get_parray_state(1, a.parent_ID)
                assert not scheduler.get_parray_state(2, a.parent_ID)
                assert scheduler.get_parray_state(3, a.parent_ID)
                assert not scheduler.get_parray_state(4, a.parent_ID)
            await ts

            ## Simple test for active task of PArray slicing.
            c = np.array([1, 2, 4, 5, 6])
            c = parray.asarray(c)

            @spawn(ts[7], placement=[cuda(1)], input=[(c[0:2], 0)])
            def task7():
                assert c[0:2].get_num_active_tasks(2) == 1
                for i in range(0, num_devices):
                    if i == 2 or i == 0:
                        assert scheduler.get_parray_state(i, c[0:2].parent_ID)
                    else:
                        assert not scheduler.get_parray_state(i, c[0:2].parent_ID)
            await ts
            
            @spawn(ts[8], placement=[cuda(3)], inout=[(c[0:2], 0)])
            def task8():
                assert c[0:2].get_num_active_tasks(4) == 1 or c[0:2].get_num_active_tasks(4) == 2
                assert c[0:2].get_num_active_tasks(2) == 1 or c[0:2].get_num_active_tasks(2) == 0
                assert scheduler.get_parray_state(2, c[0:2].parent_ID)
                assert not scheduler.get_parray_state(3, c[0:2].parent_ID)
                assert scheduler.get_parray_state(4, c[0:2].parent_ID)

            @spawn(ts[9], placement=[cuda(3)], inout=[(c[0:2], 0)])
            def task9():
                assert c[0:2].get_num_active_tasks(4) == 1 or c[0:2].get_num_active_tasks(4) == 2
                assert scheduler.get_parray_state(1, c[0:2].parent_ID)
                assert scheduler.get_parray_state(2, c[0:2].parent_ID)
                assert not scheduler.get_parray_state(3, c[0:2].parent_ID)
                assert scheduler.get_parray_state(4, c[0:2].parent_ID)
            await ts

            @spawn(ts[10], placement=[cuda(2)], inout=[(c[0:2], 0)])
            def task10():
                assert c[0:2].get_num_active_tasks(1) == 0
                assert c[0:2].get_num_active_tasks(2) == 0
                assert c[0:2].get_num_active_tasks(3) == 1 or c[0:2].get_num_active_tasks(3) == 2
                assert c[0:2].get_num_active_tasks(4) == 0 or c[0:2].get_num_active_tasks(4) == 1

                assert scheduler.get_parray_state(0, c[0:2].parent_ID)
                assert scheduler.get_parray_state(1, c[0:2].parent_ID)
                assert scheduler.get_parray_state(2, c[0:2].parent_ID)
                assert scheduler.get_parray_state(3, c[0:2].parent_ID)
                assert scheduler.get_parray_state(4, c[0:2].parent_ID)

            @spawn(ts[11], dependencies=[ts[10]], placement=[cuda(3)], input=[(c[0:2], 0)])
            def task11():
                assert c[0:2].get_num_active_tasks(1) == 0
                assert c[0:2].get_num_active_tasks(2) == 0
                assert c[0:2].get_num_active_tasks(3) == 0 or c[0:2].get_num_active_tasks(3) == 1 or c[0:2].get_num_active_tasks(3) == 2
                assert c[0:2].get_num_active_tasks(4) == 1
                assert scheduler.get_parray_state(1, c[0:2].parent_ID)
                assert scheduler.get_parray_state(2, c[0:2].parent_ID)
                assert scheduler.get_parray_state(3, c[0:2].parent_ID)
                assert scheduler.get_parray_state(4, c[0:2].parent_ID)

            @spawn(ts[12], dependencies=[ts[11]], placement=[cuda(2)], inout=[(c[0:2], 0)])
            def task12():
                assert c[0:2].get_num_active_tasks(1) == 0
                assert c[0:2].get_num_active_tasks(2) == 0
                assert c[0:2].get_num_active_tasks(3) == 1 or c[0:2].get_num_active_tasks(3) == 1
                assert c[0:2].get_num_active_tasks(4) == 0
                assert scheduler.get_parray_state(1, c[0:2].parent_ID)
                assert scheduler.get_parray_state(2, c[0:2].parent_ID)
                assert scheduler.get_parray_state(3, c[0:2].parent_ID)
                assert scheduler.get_parray_state(4, c[0:2].parent_ID)
            await ts
            # Reset devices.
            c._auto_move(-1, do_write = True)
            a._auto_move(-1, do_write = True)

            ## Simple test for PArray tracker memory management.

            # 40 bytes
            d = np.array([1, 2, 4, 5, 6])
            d = parray.asarray(d)
            @spawn(ts[13], placement=[cuda(3)], inout=[(d, 0)])
            def task13():
                assert cuda(3).query_mapped_resource(0) == 40
            await ts[13]

            assert cuda(3).query_mapped_resource(0) == 40
            d._auto_move(-1, do_write = True)
            assert cuda(3).query_mapped_resource(0) == 0

            @spawn(ts[14], placement=[cuda(2)], inout=[(d[0:2], 0)])
            def task14():
                assert cuda(2).query_mapped_resource(0) == 16
            await ts



if __name__=="__main__":
    test_parray_task()
