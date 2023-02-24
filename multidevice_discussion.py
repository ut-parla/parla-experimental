

#########
# Case A: A multidevice task launches a multi-device application. Task body is run a single time.
#########

# Comment(wlr): Case A is what I thought MultiDevice semantics are/what we are building

def worker_run():
    # Pseudo-code of Parla internals

    # Get the current task
    task = get_current_task()

    # Get the current device list
    active_devices = get_active_devices()
    stream_list = [device.get_stream() for device in active_devices]

    with create_environment(active_devices, stream_list) as e:
        # Run the task body
        task.body(e.locals)
        # Record events on the streams
        events = [stream.record_event() for stream in stream_list]
        # Pass events to dependents
        task.notify_dependents(events)


########
# Case A1: A multidevice task launches a library multi-device application
########


@spawn(gpus=2)
def multidevice_application(locals):

    # Get real device ids from local devices
    device_ids = [device.id for device in locals.devices]
    streams = [device.stream for device in locals.devices]

    # Launch a library kernel on the real devices
    # Assume A,B already distributed and in scope
    NVIDIA_GEMM_MG[device_ids, streams](A, B)


########
# Case A2: A multidevice task launches a user written multi-device application
########


@spawn(gpus=2)
def multidevice_application(locals):

    # Assume default is equipartitioning
    A = Crosspy(10, [locals.devices[0], locals.devices[1]])
    B = Crosspy(10, [locals.devices[0], locals.devices[1]])

    # Launch a user written application on the real devices
    with locals.devices[0]:
        # Executes on first device with stream0
        B[5:] = single_gpu_kernel(A[:5])

    with locals.devices[1]:
        # Executes on second device with stream1
        B[5:] = single_gpu_kernel(A[5:])


########
# Case A3: A multidevice task launches a user written multi-device application
#          Abstracted with CrossPy, assume each device only operates on predefined static partitions
########

@spawn(gpus=2)
def multidevice_application(locals):

    # Assume default is equipartitioning
    # (these could be already allocated and in scope, just an example)
    A = Crosspy(10, [locals.devices[0], locals.devices[1]])
    B = Crosspy(10, [locals.devices[0], locals.devices[1]])

    # Launch a user written application on the real devices
    # Crosspy will launch the kernel async on each device for each partition
    distributed_map(input=A, output=B, kernel=single_gpu_kernel)


#########
# Case B: A multidevice task launches a 'single-device task' for each GPU.
#         Semantics are MPI-like. Task body is run once for each device.
#########

# Comment(wlr): Case B is how I interpret hc's SIMD-like example
# Comment(wlr): Can't internally synchronize.

# Single worker version (runs in serial async loop)
# Comment(wlr): Could internally synchronize if we overload `await` to mean return to enviornment dispatch
def worker_run():
    # Pseudo-code of Parla internals

    # Get the current task
    task = get_current_task()

    # Get the current device list
    active_devices = get_active_devices()
    stream_list = [device.get_stream() for device in active_devices]

    for device, stream in *zip(active_devices, stream_list):
        with create_environment(device, stream) as e:
            # Run the task body
            task.body(e.locals)
            # Record events on the streams
            events.append(stream.record_event())
            # Pass events to dependents

    task.notify_dependents(events)


# Multiple worker version (runs in parallel, a thread per device)
# Comment(wlr): Could internally synchronize if we provide thread collective communication primitives
def task_launch():
    # Pseudo-code of Parla internals

    # Get the current task
    task = get_current_task()

    # Get the current device list
    active_devices = get_active_devices()
    stream_list = [device.get_stream() for device in active_devices]

    workgroup = create_workgroup(active_devices, stream_list, task.body)
    workgroup.notify()


def workgroup_run():

    with create_environment(device, stream) as e:
        # Run the task body
        task.body(e.locals)
        # Record events on the streams
        event = stream.record_event()

    events = workgroup.gather(from=workgroup.self, to=workgroup.master(), object=event)

    if workgroup.self == workgroup.master():
        task.notify_dependents(events)


########
# Case B1: A multidevice task launches a library multidevice kernel
########

# Comment(wlr): I don't think this is possible?

########
# Case B2: A multidevice task launches a user written multidevice kernel
#          Multidevice task is shorthand for launching many single-device tasks?
########

# Assume default is equipartitioning
A = Crosspy(10, [devices[0], devices[1]])
B = Crosspy(10, [devices[0], devices[1]])


@data(A, 0=slice(0, 5), 1=slice(5, 10))
@data(B, 0=slice(0, 5), 1=slice(5, 10))
@spawn(gpus=2)
def multidevice_application(locals):
    # Comment(wlr): I'm not sure of a nicer way to specify the index set here.
    B[locals['B'].idx] = single_gpu_kernel(A[locals['A'].idx])


########
# Case B3: A multidevice task launches a user written multidevice kernel
#          Multidevice task is shorthand for launching many single-device tasks?
#          Abstracted with CrossPy, assume task *only* operates on predefined static partitions
########
# Assume default is equipartitioning
A = Crosspy(10, [devices[0], devices[1]])
B = Crosspy(10, [devices[0], devices[1]])


@spawn(gpus=2)
def multidevice_application(locals):
    # Comment(wlr): I'm not sure of a nicer way to specify the index set here.
    B.get(locals.active_device) = single_gpu_kernel(locals.active_device)
