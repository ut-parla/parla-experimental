from parla import Parla, spawn, TaskSpace, parray
from parla.cython.device_manager import cpu, gpu
import numpy as np

# TODO(hc): This test does not exploit assert statements, but needs
#           manual checks with prints in the runtime.
#           We will expose task information to applications and
#           check correctness.

def test_parray_task():
    with Parla():
        @spawn(placement=[cpu, gpu(0)[{"vcus":0.5, "memory":1}], gpu(1), gpu(2)[{"vcus":0.3}]], vcus=1, memory=3000)
        def task():
            pass

        @spawn(placement=[gpu, cpu[{"vcus":0.4}], gpu[{"vcus":0.7, "memory":200}], gpu[{"vcus":0.3}]], vcus=1, memory=50000)
        def task2():
            pass

        @spawn(placement=[(gpu, cpu[{"vcus":0.4}], gpu[{"vcus":0.7, "memory":200}], gpu[{"vcus":0.3}])], vcus=1, memory=50000)
        def task3():
            pass

        @spawn(placement=[(gpu, cpu[{"vcus":0.4}], gpu[{"vcus":0.7, "memory":200}], gpu[{"vcus":0.3}])], vcus=1)
        def task4():
            pass

        @spawn(placement=[(gpu, cpu[{"vcus":0.4}], gpu[{"vcus":0.7, "memory":200}], gpu[{"vcus":0.3}])], memory=1000)
        def task5():
            pass

        @spawn(placement=[(gpu, cpu[{"vcus":0.4}], gpu[{"vcus":0.7, "memory":200}], gpu[{"vcus":0.3}])])
        def task6():
            pass


if __name__ == "__main__":
    test_parray_task()
