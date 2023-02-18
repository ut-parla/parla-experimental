import time
import google_benchmark as benchmark

from parla.cython import tasks as cy_tasks
from parla.common.containers import TaskSpace
flag = True
n=1000

@benchmark.register(name="Create Tasks Python")
def test(state):



    while state:

        T = TaskSpace("T")
        for i in range(n):
            task_list = T[i]

        for i in range(n):
            task_list = T[i]
        

@benchmark.register(name="Create Tasks")
def test(state):

    while state:
        T = cy_tasks.TaskSpace("T", create=flag)
        for i in range(n):
            task_list = T[i]
        
        for i in range(n):
            task_list = T[i]



@benchmark.register(name="Create Tasks Python")
def test(state):

    while state:

        T = TaskSpace("T")
        #for i in range(1000):
        #    task_list = T[i]

        #for i in range(1000):
        #    task_list = T[i]
        
        task_list = T[1:n]
        task_list = T[1:n]

@benchmark.register(name="Create Tasks, sliced")
def test(state):

    while state:
        T = cy_tasks.TaskSpace("T", shape=(n), create=flag)
        #for i in range(1000):
        #    task_list = T[i]
        
        #for i in range(1000):
        #    task_list = T[i]

        task_list = T[:]
        task_list = T[:]

        #print(task_list)


@benchmark.register(name="Create Tasks, internal")
def test(state):

    while state:
        T = cy_tasks.TaskSpace("T", create=flag)
        #for i in range(1000):
        #    task_list = T[i]
        
        #for i in range(1000):
        #    task_list = T[i]

        task_list = T[1:n]
        task_list = T[1:n]

        #print(task_list)

@benchmark.register(name="Test")
def test(state):
    while state:
        A = cy_tasks.ComputeTask()
        B = cy_tasks.ComputeTask()
        col = cy_tasks.TaskCollection([A, [B], [cy_tasks.ComputeTask(idx=i) for i in range(10)], [[cy_tasks.ComputeTask(idx=i) for i in range(n)]]])
        #print(col.tasks)


if __name__ == "__main__":
    benchmark.main()
