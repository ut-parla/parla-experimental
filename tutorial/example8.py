from parla import Parla, sleep_nogil
from parla.tasks import spawn, TaskSpace
from parla.tasks import get_current_task
from parla.devices import cpu, gpu
import time
#from parla.tasks import AtomicTaskSpace

def workload():
    active_task = get_current_task()
    print(f"Starting: {active_task}", flush=True)
    start_t = time.perf_counter()
    sleep_nogil(2000) #time to wait on cpu in microseconds
    end_t = time.perf_counter()
    print(f"Ending: {active_task}. Elapsed: {end_t - start_t} seconds", flush=True)

# Tasks MUST be defined inside a function. 
# A task created in the global scope will cause an error due to how the task closure is captured.
def main():

    @spawn(placement=cpu, vcus=0)
    async def main_task():
        T = TaskSpace("T")

        for i in range(10):
            @spawn(T[i], placement=[cpu], vcus=0.5)
            def mytask():
                workload()

        #Try converting this to the AtomicTaskSpace 
        #Is it faster? 
        await T
        #T.wait()

        print("Barrier Reached", flush=True)

        for i in range(10):
            @spawn(T[i+10], placement=[cpu], vcus=0.5)
            def mytask():
                workload()

        
    
        
if __name__ == "__main__":
     with Parla():
         main()