from parla import Parla, spawn, TaskSpace, parray
from parla.common.globals import get_scheduler, get_device_manager
from parla.cython.device_manager import cpu, gpu
import numpy as np
import crosspy as xp

PArray = parray.core.PArray

# Assume that this test uses 4 gpus.
# (Otherwise, it would be crashed)
# Each partition of the crosspy should be
# corresponding to the logical order of devices
# in @spawn's placement.
num_gpus = 4

def test_xpy_placement_from_cupy():
    import cupy as cp
    with Parla():
        t = TaskSpace("Task")
        scheduler = get_scheduler()
        device_manager = get_device_manager()

        @spawn(placement=cpu)
        async def main():
            np.random.seed(10)
            cp_list = []
            for i in range(0, num_gpus):
                with cp.cuda.Device(i):
                    cp_list.append(cp.array([1, 2, 3, 4]))
            xp_arr = xp.array(cp_list, dim=0)
            @spawn(t[0], placement=cpu, inout=[xp_arr])
            def test_cupy_task():
                for i, cp_array_list in enumerate(xp_arr.device_view()):
                    assert len(cp_array_list) == 1
                    assert i < num_gpus
                    for cp_array in cp_array_list:
                        assert isinstance(cp_array, cp.ndarray)
            await t 

            for num_arrays_for_each_partition in range(1, 3):  # The number of partitions
                cp_list = []
                for p in range(num_arrays_for_each_partition):
                    for i in range(0, num_gpus):
                        with cp.cuda.Device(i):
                            cp_list.append(cp.array([1, 2, 3, 4]))
                xp_arr = xp.array(cp_list, dim=0, wrapper=parray.asarray)
                @spawn(t[num_arrays_for_each_partition], placement=[(gpu*num_gpus)], inout=[xp_arr])
                def test_parray_task():
                    for i, cp_array_list in enumerate(xp_arr.device_view()):
                        assert len(cp_array_list) == num_arrays_for_each_partition
                        assert i < num_gpus
                        for cp_array in cp_array_list:
                            assert isinstance(cp_array, PArray)
                            assert cp_array._coherence.owner == i
                await t 
            '''
            TODO(hc): not working.
            parray_list = []
            for i in range(0, num_gpus):
                with cp.cuda.Device(i):
                    parray_list.append(parray.asarray(cp.array([1, 2, 3, 4])))
            xp_arr = xp.array(parray_list, dim=0)
            @spawn(t[1], placement=[(gpu*num_gpus)], inout=[xp_arr])
            def test_parray_task():
                for i, cp_array_list in enumerate(xp_arr.device_view()):
                    assert len(cp_array_list) == 1
                    assert i < num_gpus
                    for cp_array in cp_array_list:
                        assert isinstance(cp_array, PArray)
                        assert cp_array._coherence.owner == i
            await t 
            '''

            parray_list = []
            for i in range(0, num_gpus, 2):
                with cp.cuda.Device(i):
                    parray_list.append(parray.asarray(cp.array([1, 2, 3, 4])))

            for i in range(1, num_gpus, 2):
                with cp.cuda.Device(i):
                    parray_list.append(cp.array([1, 2, 3, 4]))

            '''
            xp_arr = xp.array(parray_list, dim=0, wrapper=parray.asarray)
            @spawn(t[1], placement=[(gpu*num_gpus)], inout=[xp_arr])
            def test_parray_task():
                for i, cp_array_list in enumerate(xp_arr.device_view()):
                    assert len(cp_array_list) == 1
                    assert i < num_gpus
                    for cp_array in cp_array_list:
                        assert isinstance(cp_array, PArray)
                        assert cp_array._coherence.owner == i
            await t 
            '''


if __name__ == "__main__":
    test_xpy_placement_from_cupy()

