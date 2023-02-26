from parla.cython import scheduler
from parla.cython import core
from parla.cython import tasks
from parla.cython import device, device_manager
from parla.utility.tracer import NVTXTracer

import inspect

from parla.cython import tasks

from typing import Union, Collection, Any

ComputeTask = tasks.ComputeTask
task_locals = tasks.task_locals
PyArchitecture = device_manager.PyArchitecture
PyDevice = device_manager.PyDevice
PlacementSource = device.PlacementSource

WorkerThread = scheduler.WorkerThread
_task_callback = scheduler._task_callback
get_scheduler_context = scheduler.get_scheduler_context

nvtx = NVTXTracer
nvtx.initialize()

Resources = core.Resources


# @profile
def _make_cell(val):
    """
    Create a new Python closure cell object.

    You should not be using this. I shouldn't be either, but I don't know a way around Python's broken semantics. (Arthur)
    """
    x = val

    def closure():
        return x

    return closure.__closure__[0]


# @profile
def spawn(task=None,
          dependencies=[],
          # TODO(hc): Do we support TaskID? (IIRC, it will be removed?)
          placement: Collection[Union[Collection[PlacementSource], Any, None]] = None,
          vcus=1):
    nvtx.push_range(message="Spawn::spawn", domain="launch", color="blue")

    scheduler = get_scheduler_context().scheduler

    if not isinstance(task, tasks.Task):
        taskspace = scheduler.default_taskspace

        if task is None:
            idx = len(taskspace)

        task = taskspace[idx]

    # @profile
    def decorator(body):
        nonlocal vcus
        nonlocal dependencies
        nonlocal task
        nonlocal placement

        if inspect.iscoroutine(body):
            separated_body = body
        else:
            separated_body = type(body)(
                body.__code__, body.__globals__, body.__name__, body.__defaults__,
                closure=body.__closure__ and tuple(_make_cell(x.cell_contents) for x in body.__closure__))
            separated_body.__annotations__ = body.__annotations__
            separated_body.__doc__ = body.__doc__
            separated_body.__kwdefaults__ = body.__kwdefaults__
            separated_body.__module__ = body.__module__

        flattened_dependencies = []
        tasks.flatten_tasks(dependencies, flattened_dependencies)

        scheduler = get_scheduler_context().scheduler

        # Get a set of candidate devices for a task.
        device_reqs = scheduler.get_device_reqs_from_placement(placement)
        task.set_device_reqs(device_reqs)

        task.set_scheduler(scheduler)
        task.instantiate(function=_task_callback,
                         args=(separated_body,),
                         dependencies=flattened_dependencies,
                         constraints=vcus)

        scheduler.spawn_task(task)
        # scheduler.run_scheduler()
        nvtx.pop_range(domain="launch")

        # This is a complete hack but somehow performs better than doing the "right" thing of signaling from waiting threads that the compute bound thread needs to release the GIL.
        # TODO: Make this an optional flag.
        if ((task_locals.spawn_count % 10 == 0)):
            scheduler.spawn_wait()
        task_locals.spawn_count += 1

        return task

    return decorator
