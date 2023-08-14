from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu, gpu
import time

# Tasks MUST be defined inside a function. 
# A task created in the global scope will cause an error due to how the task closure is captured.
def main():

    @spawn(placement=cpu, vcus=0)
    async def main_task():
        T = TaskSpace("T")

        @spawn(T[0], placement=[gpu])
        def mytask():
            print(f"Hi I'm {T[0]}. I'm currently {T[0].state}.", flush=True)

        @spawn(T[1], placement=[cpu], dependencies=[T[0]])
        def mytask2():
            print(f"Hi I'm {T[1]}. I'm currently {T[1].state}.", flush=True)
        
if __name__ == "__main__":
     with Parla():
         main()