from parla import Parla
from parla.tasks import spawn, TaskSpace, get_scheduler
from parla.common.globals import get_active_device
from parla.devices import cpu, gpu

import numpy as np
from parla.array import asarray


# def test_parray_task():
#     print("test_parray_task", flush=True)

#     with Parla():
#         @spawn(placement=cpu)
#         async def main():
#             A = np.array([1, 2, 3], dtype=np.int32)
#             A = asarray(A)
#             T = TaskSpace("T")

#             @spawn(T[0], placement=[gpu(0)], input=[A])
#             def task():
#                 scheduler = get_scheduler()
#                 device_id = 1

#                 # CHECK GPU MEMORY
#                 max_memory = scheduler.get_max_memory(device_id)
#                 mapped_memory = scheduler.get_mapped_memory(device_id)

#                 assert mapped_memory == 12

#                 reserved_memory = scheduler.get_reserved_memory(device_id)
#                 assert reserved_memory == max_memory - 12

#                 # CHECK CPU MEMORY
#                 device_id = 0
#                 max_memory = scheduler.get_max_memory(device_id)
#                 mapped_memory = scheduler.get_mapped_memory(device_id)

#                 assert mapped_memory == 12

#                 reserved_memory = scheduler.get_reserved_memory(device_id)
#                 assert reserved_memory == max_memory - 12

#             @spawn(T[1], placement=[cpu], dependencies=[T[0]], inout=[A])
#             def task():
#                 scheduler = get_scheduler()
#                 device_id = 1

#                 # CHECK GPU MEMORY
#                 max_memory = scheduler.get_max_memory(device_id)
#                 mapped_memory = scheduler.get_mapped_memory(device_id)

#                 assert mapped_memory == 0

#                 reserved_memory = scheduler.get_reserved_memory(device_id)
#                 assert reserved_memory == max_memory

#                 # CHECK CPU MEMORY
#                 device_id = 0
#                 max_memory = scheduler.get_max_memory(device_id)
#                 mapped_memory = scheduler.get_mapped_memory(device_id)

#                 assert mapped_memory == 12

#                 reserved_memory = scheduler.get_reserved_memory(device_id)
#                 assert reserved_memory == max_memory - 12


def test_internal_alloc():
    print("test_internal_alloc", flush=True)
    import cupy as cp

    with Parla():

        @spawn(placement=cpu)
        def main():

            @spawn(placement=[gpu(0)], memory=40)
            def task():

                scheduler = get_scheduler()
                device_id = 1

                # CHECK GPU MEMORY
                max_memory = scheduler.get_max_memory(device_id)
                mapped_memory = scheduler.get_mapped_memory(device_id)

                assert mapped_memory == 40

                reserved_memory = scheduler.get_reserved_memory(device_id)
                assert reserved_memory == max_memory - 40

                A = cp.asarray(cp.arange(3), dtype=np.int32)
                A = asarray(A)

                # CHECK GPU MEMORY
                max_memory = scheduler.get_max_memory(device_id)
                mapped_memory = scheduler.get_mapped_memory(device_id)

                assert mapped_memory == 40

                reserved_memory = scheduler.get_reserved_memory(device_id)
                assert reserved_memory == max_memory - 40


def test_nested_chain():
    import cupy as cp

    ntasks = 10

    with Parla():

        def generated_nested(i=0, B=None):

            if i > ntasks:
                return

            if B is None:
                read = []
            else:
                read = [B]

            @spawn(placement=[gpu(0)], memory=0, input=read)
            def task():

                A = cp.asarray(cp.arange(3), dtype=np.int32)
                A = asarray(A)

                scheduler = get_scheduler()

                mapped_memory = scheduler.get_mapped_memory(1)
                assert mapped_memory == 12*(i+1)

                max_memory = scheduler.get_max_memory(1)
                reserved_memory = scheduler.get_reserved_memory(1)
                assert reserved_memory == max_memory - 12*(i+1)

                generated_nested(i + 1, A)

                mapped_memory = scheduler.get_mapped_memory(1)
                assert mapped_memory == 12*(i+1)

                reserved_memory = scheduler.get_reserved_memory(1)
                assert reserved_memory == max_memory - 12*(i+1)

        generated_nested()


def test_nested_chain_internal():
    print("test_nested_chain_internal", flush=True)
    import cupy as cp

    ntasks = 10

    with Parla():

        T = TaskSpace("T")

        def generated_nested(i=0, B=None):

            if i > ntasks:
                return

            if B is None:
                read = []
            else:
                read = [B]

            @spawn(T[i], placement=[gpu(0)], dependencies=[T[i-1]], memory=40, input=read)
            def task():

                A = cp.asarray(cp.arange(3), dtype=np.int32)
                A = asarray(A)

                scheduler = get_scheduler()
                device_id = 1

                mapped_memory = scheduler.get_mapped_memory(1)
                max_memory = scheduler.get_max_memory(1)
                reserved_memory = scheduler.get_reserved_memory(1)

                assert mapped_memory == 40 + 12*(i)
                assert reserved_memory == max_memory - (40 + 12*(i))

                # print(i, max_memory - reserved_memory)
                # print(i, mapped_memory)

                generated_nested(i + 1, A)

                mapped_memory = scheduler.get_mapped_memory(1)

                assert reserved_memory == max_memory - (40 + 12*(i))
                # print(i, max_memory - reserved_memory)
                # print(i, mapped_memory)

        generated_nested()
