from parla import Parla, spawn, TaskSpace, sleep_nogil
# from sleep.core import bsleep

bsleep = sleep_nogil

def main(T):


    @spawn(T[0], vcus=0)
    def task1():
        print("+HELLO OUTER 0", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 0", flush=True)

    @spawn(T[1], vcus=0, dependencies=[T[0]])
    def task2():
        print("+HELLO OUTER 1", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 1", flush=True)

    @spawn(T[2], vcus=0)
    def task3():
        print("+HELLO OUTER 2", flush=True)
        bsleep(1000)
        print("-HELLO OUTER 2", flush=True)

    # @spawn()
    # def test():
    #    print("HELLO", flush=True)


if __name__ == "__main__":
    with Parla():
        T = TaskSpace("T")
        main(T)
