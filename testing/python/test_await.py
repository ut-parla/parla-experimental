from parla import Parla, spawn
from parla.tasks import AtomicTaskSpace as TaskSpace
from parla import sleep_nogil


def wrapper_chain(T, i=0):
    @spawn(T[i], dependencies=[T[i - 1]])
    async def outer():
        TT = TaskSpace(f"task_{i}")
        await TT


def test_chain():
    with Parla():
        T = TaskSpace("main")
        i = 0
        while i < 30000:
            wrapper_chain(T, i)
            i += 1


if __name__ == "__main__":
    test_chain()
