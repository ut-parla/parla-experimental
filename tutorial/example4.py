from parla import Parla, sleep_nogil
from parla.tasks import spawn, TaskSpace
from parla.tasks import get_current_task
from parla.devices import cpu, gpu
import time


def workload():
    active_task = get_current_task()
    print(f"Starting: {active_task}", flush=True)
    start_t = time.perf_counter()
    sleep_nogil(2000)
    end_t = time.perf_counter()
    print(f"Ending: {active_task}. Elapsed: {end_t - start_t} seconds", flush=True)

# Tasks MUST be defined inside a function. 
# A task created in the global scope will cause an error due to how the task closure is captured.
def main():

    @spawn(placement=cpu, vcus=0)
    async def main_task():
        T = TaskSpace("T")
        load = 0.5

        #Try running where both tasks can't fit on the same device
        #load = 0.75

        @spawn(T[1], placement=[cpu], vcus=load)
        def mytask2():
            workload()

        @spawn(T[0], placement=[cpu], vcus=load)
        def mytask():
            workload()

        
if __name__ == "__main__":
     with Parla():
         main()