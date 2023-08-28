from parla import Parla, sleep_nogil
from parla.tasks import spawn, TaskSpace
from parla.tasks import get_current_task
from parla.devices import cpu, gpu
import time


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

        #THIS EXAMPLE IS BROKEN ON PURPOSE
        #Task 5 is not created
        #It will hang forever waiting for a task that will never be created

        #Task 11 should depend on all previous tasks. 
        #Fix this with sparse access slicing notation

        for i in range(10):
            if i == 5:
                continue
            @spawn(T[i], placement=[cpu], vcus=0.5)
            def mytask():
                workload()

        @spawn(T[11], placement=[cpu], vcus=0.5, dependencies=[T[0:10]])
        def mytask():
            workload()
    
        
if __name__ == "__main__":
     with Parla():
         main()