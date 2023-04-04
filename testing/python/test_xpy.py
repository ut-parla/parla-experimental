from parla import Parla, spawn, TaskSpace, parray
from parla.common.globals import get_scheduler, get_device_manager
from parla.cython.device_manager import cpu, gpu
import numpy as np
import crosspy as xp

# Assume that this test uses 4 gpus.
# (Otherwise, it would be crashed)
# Each partition of the crosspy should be
# corresponding to the logical order of devices
# in @spawn's placement.
num_gpus = 4

def test_xpy_placement_from_numpy():
    with Parla():
        t = TaskSpace("Task")
        scheduler = get_scheduler()
        device_manager = get_device_manager()

        @spawn(placement=cpu)
        async def main():
            np.random.seed(10)
            np_list = []
            for i in range(0, num_gpus):
                np_list.append(np.array([1, 2, 3, 4]))
            xp_arr = xp.array(np_list, dim=0)
            '''
            @spawn(t[0], placement=cpu, inout=[xp_arr])
            def test_numpy_task():
                for i, np_array_list in enumerate(xp_arr.device_view()):
                    assert len(np_array_list) == 1
                    assert i < num_gpus
                    for np_array in np_array_list:
                        assert isinstance(np_array, np.ndarray)
            await t 
            '''


if __name__ == "__main__":
    test_xpy_placement_from_numpy()

