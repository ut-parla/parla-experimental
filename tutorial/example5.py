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

        #Accessing a TaskSpace creates a task handle
        print(f"The TaskSpace Before: {T}")
        print(T[0], T[1])
        print(f"The TaskSpace After: {T}")

        
if __name__ == "__main__":
     with Parla():
         main()