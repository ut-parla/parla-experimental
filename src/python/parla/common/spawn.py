from parla.common import containers
from parla.cython import scheduler
from parla.cython import core

import inspect

from parla.cython import tasks

import nvtx

TaskID = tasks.TaskID
task_locals = tasks.task_locals

WorkerThread = scheduler.WorkerThread
_task_callback = scheduler._task_callback
get_scheduler_context = scheduler.get_scheduler_context

Tasks = containers.Tasks

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
def spawn(taskid=None,  dependencies=[], vcus=1):
    nvtx.push_range(message="Spawn::spawn", domain="launch", color="blue")
    if not taskid:
        taskid = TaskID("global_" + str(len(task_locals.global_tasks)),
                        (len(task_locals.global_tasks),))

        task_locals.global_tasks += [taskid]

    # @profile
    def decorator(body):
        nonlocal vcus
        req = Resources(vcus)

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

        taskid.dependencies = dependencies
        processed_dependencies = Tasks(*dependencies)._flat_tasks

        scheduler = get_scheduler_context().scheduler

        task = scheduler.spawn_task(function=_task_callback, args=(separated_body,),
                                    dependencies=processed_dependencies, taskid=taskid,
                                    req=req, name=getattr(body, "___name__", None))
        # scheduler.run_scheduler()
        nvtx.pop_range(domain="launch")

        scheduler.spawn_wait()

        return task

    return decorator
