from parla import Parla, spawn, TaskSpace

def test_duplicate_task():
    with Parla():
        @spawn()
        def main():
            T = TaskSpace("T")

            A = T[1]
            B = T[1]
            @spawn(A)
            def task_funcA():
                for i in range(10):
                    print(i, flush=True)
            try: 
                @spawn(B)
                def task_funcB():
                    for i in range(10):
                        print(i, flush=True)
            except Exception as excinfo:
                assert True
                assert str(excinfo) == "Duplicate task ID spawned. This will cause runtime to hang. Aborting..."

test_duplicate_task()