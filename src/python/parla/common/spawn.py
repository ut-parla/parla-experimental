"""!
@file spawn.py
@brief Contains the core user-facing API to spawn tasks.
"""

from __future__ import annotations  # For type hints of unloaded classes
from parla.cython import scheduler
from parla.cython import core
from parla.cython import tasks
from parla.cython import device, device_manager
from parla.common.dataflow import Dataflow
from parla.common.parray.core import PArray
from parla.utility.tracer import NVTXTracer
from parla.common.globals import (
    default_sync,
    VCU_BASELINE,
    SynchronizationType,
    crosspy,
    CROSSPY_ENABLED,
)
import inspect
from typing import Collection, Any, Union, List, Tuple

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
def spawn(
    task=None,
    dependencies=[],
    # This collection does not contain Union anymore, which was used by the
    # old Parla, since we now allow support {arch, arch, arch} placement
    # to map a task to three devices.
    placement: Collection[Union[Collection[PlacementSource], Any, None]] = None,
    # TODO(hc): this will be refined to support multi-dimensional CrossPy
    #           support
    input: List[Union[crosspy.CrossPyArray, Tuple[PArray, int]]] = None,
    output: List[Union[crosspy.CrossPyArray, Tuple[PArray, int]]] = None,
    inout: List[Union[crosspy.CrossPyArray, Tuple[PArray, int]]] = None,
    vcus: float = None,
    memory: int = None,
    runahead: SynchronizationType = default_sync,
):
    nvtx.push_range(message="Spawn::spawn", domain="launch", color="blue")
    scheduler = get_scheduler_context().scheduler
    if not isinstance(task, tasks.Task):
        taskspace = scheduler.default_taskspace

        if task is None:
            idx = len(taskspace)
        else:
            idx = task

        task = taskspace[idx]

    # @profile
    def decorator(body):
        nonlocal vcus
        nonlocal memory
        nonlocal dependencies
        nonlocal task
        nonlocal placement
        nonlocal runahead

        # COMMENT(wlr): Just added this back to revert my commit 30 min ago.
        if vcus is not None:
            # Default behavior the same as Parla 0.2.
            # The baseline should not be multiplied when placement is None.
            # This is because architecture's __getitem__ also multiplies
            # the baseline, and it multiplies (baseline^2).
            if vcus <= 1 and placement is not None:
                vcus = int(vcus * VCU_BASELINE)
            else:
                # Only large values for ease of testing
                vcus = int(vcus)
        if memory is not None:
            memory = int(memory)

        if inspect.iscoroutine(body):
            separated_body = body
        else:
            separated_body = type(body)(
                body.__code__,
                body.__globals__,
                body.__name__,
                body.__defaults__,
                closure=body.__closure__
                and tuple(_make_cell(x.cell_contents) for x in body.__closure__),
            )
            separated_body.__annotations__ = body.__annotations__
            separated_body.__doc__ = body.__doc__
            separated_body.__kwdefaults__ = body.__kwdefaults__
            separated_body.__module__ = body.__module__

        flattened_dependencies = []
        tasks.flatten_tasks(dependencies, flattened_dependencies)

        scheduler = get_scheduler_context().scheduler

        device_manager = scheduler.device_manager

        # Get a set of candidate devices for a task.
        # If none of the placement is passed, make
        # all devices candidate.
        placement = (
            placement
            if placement is not None
            else [
                arch[
                    {
                        "vcus": vcus if vcus is not None else 0,
                        "memory": memory if memory is not None else 0,
                    }
                ]
                for arch in device_manager.get_all_architectures()
            ]
        )

        # print("placement: ", placement)

        device_reqs = scheduler.get_device_reqs_from_placement(placement, vcus, memory)
        task.set_device_reqs(device_reqs)

        dataflow = Dataflow(input, output, inout)
        task.set_scheduler(scheduler)
        task.instantiate(
            function=_task_callback,
            args=(separated_body,),
            dependencies=flattened_dependencies,
            dataflow=dataflow,
            runahead=runahead,
        )
        try:
            scheduler.spawn_task(task)
        except RuntimeError:
            raise RuntimeError(
                "Conflicting task state while spawning task. Possible duplicate TaskID: "
                + str(task)
            )
        # scheduler.run_scheduler()
        nvtx.pop_range(domain="launch")

        # This is a complete hack but somehow performs better than doing the "right" thing of signaling from waiting threads that the compute bound thread needs to release the GIL.
        # TODO: Make this an optional flag.
        if task_locals.spawn_count % 10 == 0:
            scheduler.spawn_wait()
        task_locals.spawn_count += 1

        return task

    return decorator
