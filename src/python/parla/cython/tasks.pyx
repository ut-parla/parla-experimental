
from collections import namedtuple

cimport core
from parla.cython import core
from abc import abstractmethod, ABCMeta
from typing import Optional, List
import threading

import traceback
import sys

import nvtx

class TaskState(object, metaclass=ABCMeta):
    __slots__ = []

    @property
    @abstractmethod
    def value(self) -> int:
        raise NotImplementedError()
        
    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError()


class TaskCreated(TaskState):
    """
    This state specifies that a task is waiting for dependencies' spawnings
    """
    @property
    def value(self):
        return 0

    @property
    def is_terminal(self):
        return False


class TaskSpawned(TaskState):
    """
    This state specifies that a task is ready to be mapped to a specific device set for execution
    """
    @property
    def value(self):
        return 1

    @property
    def is_terminal(self):
        return False


class TaskMapped(TaskState):
    """
    This state specifies that a task has been mapped to a device set.
    """
    @property
    def value(self):
        return 2

    @property
    def is_terminal(self):
        return False


class TaskReserved(TaskState):
    """
    This state specifies that a task has had its persistent resources (e.g. memory) reserved on its device set
    """
    @property
    def value(self):
        return 3

    @property
    def is_terminal(self):
        return False


class TaskReady(TaskState):
    """
    This state specifies that a task is "ready" to be launched. Its dependencies have been dispatched to hardware queues (or have completed)
    """
    @property
    def value(self):
        return 4

    @property
    def is_terminal(self):
        return False


class TaskRunning(TaskState):
    __slots__ = ["func", "args", "dependencies"]

    @property
    def value(self):
        return 5

    @property
    def is_terminal(self):
        return False

    # The argument dependencies intentially has no type hint.
    # Callers can pass None if they want to pass empty dependencies.
    def __init__(self, func, args, dependencies: Optional[List]):
        #print("TaskRunning init", flush=True)
        if dependencies is not None:
            # d could be one of four types: Task, DataMovementTask, TaskID or other types.
            #assert all(isinstance(d, (Task, TaskID)) for d in dependencies)
            #self.dependencies = [
            #    d for d in dependencies if isinstance(d, Task)]

            #COMMENT(wlr): I think we shouldn't filter out the TaskID here. Otherwise, we cannot barrier on unspawned tasks
            self.dependencies = dependencies
        else:
            self.dependencies = []

        self.args = args
        self.func = func
        #print("TaskRunning init done", flush=True)

    def clear_dependencies(self):
        self.dependencies = []

    def __repr__(self):
        if self.func:
            # return "TaskRunning({}, {}, {})".format(self.func.__name__, self.args, self.dependencies)
            return "TaskRunning({})".format(self.func.__name__)
        else:
            return "Functionless task"

class TaskCompleted(TaskState):
    __slots__ = ["return_value"]

    @property
    def value(self):
        return 6

    def __init__(self, ret):
        self.return_value = ret

    @property
    def is_terminal(self):
        return True

    def __repr__(self):
        return "TaskCompleted({})".format(self.return_value)


class TaskException(TaskState):
    __slots__ = ["exception", "traceback"]

    @property
    def value(self):
        return 7

    @property
    def is_terminal(self):
        return True

    def __init__(self, exc=None, tb=None):
        self.exception = exc
        self.traceback = tb

    def __repr__(self):
        return "TaskException({})".format(self.exception)

TaskAwaitTasks = namedtuple("AwaitTasks", ["dependencies", "value_task"])


class _TaskLocals(threading.local):
    def __init__(self):
        super(_TaskLocals, self).__init__()
        self.task_scopes = []
        self.spawn_count = 0

    @property
    def ctx(self):
        return getattr(self, "_ctx", None)

    @ctx.setter
    def ctx(self, v):
        self._ctx = v

    @property
    def global_tasks(self):
        return getattr(self, "_global_tasks", [])

    @global_tasks.setter
    def global_tasks(self, v):
        self._global_tasks = v


task_locals = _TaskLocals()

class TaskID:

    def __init__(self, name, id, taskspace):
        self._name = name
        self._id = id
        self._task = None
        self._taskspace = None

    @property
    def task(self):
        if not self._task:
            return None
        return self._task

    @property
    def inner_task(self):
        if not self._task:
            return None
        return self._task.inner_task

    @task.setter
    def task(self, v):
        assert not self._task
        self._task = v

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def full_name(self):
        return "_".join(str(i) for i in (self._name, *self._id))

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, v):
        self._dependencies = v

    def __hash__(self):
        return hash(self.full_name)

    def __await__(self):
        return (yield TaskAwaitTasks([self.task], self.task))

    def __dealloc__(self):
        core.binlog_0("Task", "TaskID {} is being deallocated".format(self.full_name))


class Task:

    def __init__(self, dependencies=None, taskid=None, req=0, name="default_name", state=TaskCreated(), assigned=False, scheduler=None):

        self.id = id(self)
        self.taskid = taskid

        #TODO: Fix resources
        if isinstance(req, core.Resources):
            self.req = req
        else:
            self.req = core.Resources(req)
            
        self.name = name
        self.assigned = assigned

        self.inner_task = core.PyInnerTask(self.id, self, self.req.resources)
        self.scheduler = scheduler

        self.dependencies  = dependencies
        self.state = state

    def _wait_for_dependency_events(self, enviornment):
        pass

    @property
    def result(self):

        if isinstance(self.state, TaskCompleted):
            return self.state.return_value
        elif isinstance(self.state, TaskException):
            return self.state.exception

        return None

    @abstractmethod
    def _execute_task(self):
        raise NotImplementedError()
    
    @abstractmethod
    def _finish(self, ctx):
        raise NotImplementedError()

    def run(self):
        #assert self.assigned, "Task was not assigned to a device before execution"
        #assert isinstance(self.req, EnvironmentRequirements), "Task was not assigned to a enviornment before execution"

        task_state = None
        try:
            #assert(self._state, TaskRunning)

            #TODO: Load the environment
            #TODO: Get the events

            task_state = self._execute_task()

            #TODO: Record & Sync Events

            task_state = task_state or TaskCompleted(None)

        except Exception as e:
            tb = traceback.format_exc()
            task_state = TaskException(e, tb)

            if isinstance(e, KeyboardInterrupt):
                print("You pressed Ctrl+C! In a Task!", flush=True)
                raise e
            #print("Task {} failed with exception: {} \n {}".format(self.name, e, tb), flush=True)

        finally:
            assert(task_state is not None)
            self.state = task_state

    def __await__(self):
        return (yield TaskAwaitTasks([self], self))

    def add_dependencies(self, dependency_list, process=True):
        return self.inner_task.add_dependencies(dependency_list, process)

    def get_num_dependencies(self):
        return self.inner_task.get_num_dependencies()

    def get_num_dependents(self):
        return self.inner_task.get_num_dependents()

    def get_num_blocking_dependencies(self):
        return self.inner_task.get_num_blocking_dependencies()

    def get_dependencies(self):
        dependency_list = self.inner_task.get_dependencies()
        return dependency_list

    def get_dependents(self):
        dependent_list = self.inner_task.get_dependents()
        return dependent_list

    def notify_dependents_wrapper(self):
        """ Mock interface only used for testing. Notify dependents should be called internall by the scheduler """
        status = self.inner_task.notify_dependents_wrapper()
        return status

    def set_state(self, state):
        self.inner_task.set_state(state)

    def get_state(self):
        return self.inner_task.get_state()

    def set_complete(self):
        self.inner_task.set_complete()

    def __repr__(self):
        return "Task. {}".format(self.taskid.full_name)

    def __dealloc__(self):
        print("Task Deallocation")


class ComputeTask(Task):

    def __init__(self, func, args, dependencies=None, taskid=None, req=None, name="default_name", state=TaskCreated(), assigned=False, scheduler=None, dataflow=None):
        super().__init__(dependencies, taskid, req, name, state, assigned, scheduler)
        self.func = func
        self.args = args
        self.dataflow = dataflow # input/output/inout of the task


        # Expose the taskid->task reference to other threads as late as possible,
        # but not after potentially getting scheduled.
        taskid.task = self

    def _execute_task(self):
        return self.func(self, *self.args)

    def cleanup(self):
        self.func = None
        self.args = None
        self.dataflow = None

    def _finish(self, context):
        pass

    def __dealloc__(self):
        print("ComputeTask dealloc", self.name, flush=True)


#TODO: Data Movement Task  


