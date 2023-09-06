from parla import Parla, spawn, TaskSpace


def test_default_start():
    with Parla():
        @spawn()
        def main():
            n = 10
            l = n//2

            T = TaskSpace("T")
            A = T[1]
            B = T[1]

            @spawn(A)
            def task_func():
                for i in range(10):
                    print(i)
            # print("Starting B")
            # @spawn(B)
            # def task_funcB():
            #     print("In B")
            #     for i in range(10):
            #         print(i)

test_default_start()