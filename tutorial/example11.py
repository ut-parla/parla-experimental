from parla import Parla, sleep_nogil
from parla.tasks import spawn, TaskSpace
from parla.tasks import get_current_task, get_current_context
from parla.tasks import specialize
from parla.devices import cpu, gpu
import time
#from parla.tasks import AtomicTaskSpace


@specialize
def variant_function():
    print("Default Variant Implementation", flush=True)

#Define the variant for any number of CPUs
@variant_function.variant(architecture=cpu)
def cpu_variant_function():
    print("CPU Variant Implementation", flush=True)

#Define the variant for any number of GPUs
@variant_function.variant(architecture=gpu)
def gpu_variant_function():
    print("GPU Variant Implementation", flush=True)


#Define the variant for 2 GPUs
# @variant_function.variant(gpu*2)
# def gpu_2_variant_function():
#     print("2 GPU Variant Implementation", flush=True)

def workload():
    active_task = get_current_task()
    context = get_current_context()

    #Serial Loop over devices
    for device in context.devices:
       print(f"Active Device: {device}", flush=True)

    # Parallel Loop over devices (launches an internal thread for each device)
    #@context.parfor()
    #def device_work(context):
    #    print(f"Active Device: {context.device}", flush=True)

    

# Tasks MUST be defined inside a function. 
# A task created in the global scope will cause an error due to how the task closure is captured.
def main():

    @spawn(placement=cpu, vcus=0)
    async def main_task():
        T = TaskSpace("T")

        for i in range(1):
            @spawn(T[i], placement=[(cpu, gpu)])
            def mytask():
                workload()

        
    
        
if __name__ == "__main__":
     with Parla():
         main()