from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu
from typing import Callable, Optional, List
from time import perf_counter
import numpy as np


def run(function: Callable[[], Optional[TaskSpace]]):
    """
    Parla applications are typically run in a top-level task.
    This function encapsulates the creation of the top-level task and the Parla runtime.

    Args:
        function (Callable[[], Optional[TaskSpace]]): A parla app that (optionally) returns a task space.
    """
    # Start the Parla runtime
    with Parla():
        # Create an encapsulating top-level task to kick off the computation and wait for it to complete.
        @spawn(placement=cpu, vcus=0)
        async def top_level_task():
            # Run the Parla application
            await function()


async def simple_barrier_func():
    T = TaskSpace("T")

    i = 0

    @spawn(T[i])
    def task1():
        print(f"Hello from task {i}!", flush=True)
        return np.ones(4) + i

    i = 1

    @spawn(T[i])
    def task2():
        print(f"Hello from task {i}!", flush=True)
        return np.ones(4) + i

    A = await T[0]
    B = await T[1]
    assert np.all(A == 1), f"{A} != 1"
    assert np.all(B == 2), f"{B} != 2"


def main():

    @spawn(placement=cpu, vcus=0)
    async def simple_barrier_task():
        T = TaskSpace("T")

        i = 0

        @spawn(T[i])
        def task1():
            print(f"Hello from task {i}!", flush=True)
            return np.ones(4) + i

        i = 1

        @spawn(T[i])
        def task2():
            print(f"Hello from task {i}!", flush=True)
            return np.ones(4) + i

        A = await T[0]
        B = await T[1]
        print(A)
        print(B)
        assert np.all(A == 1), f"{A} != 1"
        assert np.all(B == 2), f"{B} != 2"


def test_task_return():
    run(simple_barrier_func)


def test_task_return_main():
    with Parla():
        main()


if __name__ == "__main__":
    test_task_return_main()
