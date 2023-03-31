from parla import Parla, spawn, TaskSpace, parray
from parla.common.globals import get_scheduler, get_device_manager, get_current_context, get_current_devices
from parla.cython.device_manager import cpu, cuda

import numpy as np

def test_task_mapping_policy():
    with Parla():
        @spawn(placement=[cpu])
        async def main():
            a = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            b = np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]])
            a = parray.asarray(a)
            b = parray.asarray(b)
            ts = TaskSpace("Mapping")
            scheduler = get_scheduler()

            # Single device task mapping test.

            @spawn(ts[0], placement=[cuda(1)], input=[(a, 0)])
            def t0():
                ctx = get_current_context()
                assert ctx().get_global_id() == 2
                print("t0 done\n", flush=True)

            @spawn(ts[1], placement=[cuda(0), cuda(1), cuda(2), cuda(3)], dependencies=[ts[0]], input=[(a, 0)])
            def t1():
                ctx = get_current_context()
                assert ctx().get_global_id() == 2
                print("t1 done\n", flush=True)

            @spawn(ts[2], placement=[cuda(2)], dependencies=[ts[1]], input=[(a, 0)])
            def t2():
                ctx = get_current_context()
                assert ctx().get_global_id() == 3
                print("t2 done\n", flush=True)

            await ts

            # Multi-device task mapping test.

            @spawn(ts[3], placement=[cuda(3)], input=[(b, 0)])
            def t3():
                ctx = get_current_context()
                assert ctx().get_global_id() == 4
                print("t3 done\n", flush=True)

            @spawn(ts[4], placement=[(cuda(1), cuda(2)), (cuda(1), cuda(4)), (cuda(0), cuda(1)), (cpu(0), cuda(1)), (cuda(1), cuda(3))], dependencies=[ts[3]], input=[(a, 0), (b, 1)])
            def t4():
                devs = get_current_devices()
                assert devs[0]().get_global_id() == 2
                assert devs[1]().get_global_id() == 4
                print("t4 done\n", flush=True)

            await ts

            # Single architecture task mapping task.

            @spawn(ts[5], placement=[cuda], input=[(b, 0)])
            def t5():
                ctx = get_current_context()
                assert ctx().get_global_id() == 4
                print("t5 done\n", flush=True)

            # 1, 2, -1
            @spawn(ts[6], placement=[cuda], input=[(a, 0)])
            def t6():
                ctx = get_current_context()
                assert ctx().get_global_id() == 2 or ctx().get_global_id() == 3
                assert not ctx().get_global_id() == 0
                assert not ctx().get_global_id() == 1
                assert not ctx().get_global_id() == 4
                print("t6 done\n", flush=True)

            # [Slice test]

            c = np.array([1, 2, 4, 5, 6, 1, 2, 4, 5, 6, 1, 2, 4, 5, 6])
            c = parray.asarray(c)

            @spawn(ts[7], placement=[cuda(0)], input=[(c[0:4], 0)])
            def t7():
                ctx = get_current_context()
                assert ctx().get_global_id() == 1
                print("t7 done\n", flush=True)

            @spawn(ts[8], placement=[cuda(1)], input=[(c[0:2], 0)])
            def t8():
                ctx = get_current_context()
                assert ctx().get_global_id() == 2
                print("t8 done\n", flush=True)

            @spawn(ts[9], placement=[cuda(2)], input=[(c[6:9], 0)])
            def t9():
                ctx = get_current_context()
                assert ctx().get_global_id() == 3
                print("t9 done\n", flush=True)
            await ts

            # When a policy calculate locality score, it can access root PArray information.
            # It does not calculate subarray's information.

            # Input should be slices before they are write back  

            # Multi-device placements consisting of the same devices are never chosen.
            @spawn(ts[10], placement=[(cuda(0), cuda(3)), (cuda(1), cuda(3))], inout=[(c,0), (c,1)], dependencies=[ts[9]])
            def t10():
                print("t10 start\n", flush=True)
                devs = get_current_devices()
                assert devs[0]().get_global_id() == 1 or devs[0]().get_global_id() == 2 or devs[0]().get_global_id() == 4
                assert not devs[1]().get_global_id() == 1 and (devs[1]().get_global_id() == 4)
                print("t10 done\n", flush=True)
            await ts

            '''
            @spawn(ts[11], placement=[(cuda(1), cuda(0)), (cuda(0), cuda(3)), (cuda(1), cuda(3))], inout=[(c, 0), (c[0:1], 1)], dependencies=[ts[10]])
            def t11():
                devs = get_current_devices()
                assert devs[0]().get_global_id() == 2
                assert devs[1]().get_global_id() == 1

            # after this, c slice is evicted from cuda(2). 
            await ts

            @spawn(ts[12], placement=[(cuda(0), cuda(1)), (cuda(0), cuda(2))], output=[(c, 0), (c, 1)])
            def t12():
                devs = get_current_devices()
                assert devs[0]().get_global_id() == 1
                assert devs[1]().get_global_id() == 2
            '''


if __name__ == "__main__":
    test_task_mapping_policy()
