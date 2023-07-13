from parla import Parla, sleep_nogil
from parla.tasks import spawn, TaskSpace
from parla.tasks import get_current_task, get_current_context
from parla.devices import cpu, gpu
import time
#from parla.tasks import AtomicTaskSpace

def workload():
    active_task = get_current_task()
    context = get_current_context()
    print(f"Running: {active_task} on {context}", flush=True)


# Tasks MUST be defined inside a function. 
# A task created in the global scope will cause an error due to how the task closure is captured.
def main():

    @spawn(placement=cpu, vcus=0)
    async def main_task():
        T = TaskSpace("T")

        for i in range(10):
            #This task will be placed on either the CPU or any GPU
            #Obverve the output to see where it is placed
            @spawn(T[i], placement=[gpu, cpu], vcus=0.5)
            def mytask():
                workload()

        
    
        
if __name__ == "__main__":
     with Parla():
         main()