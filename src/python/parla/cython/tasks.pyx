
from collections import namedtuple

cimport core
from parla.cython import core
from abc import abstractmethod, ABCMeta
from typing import Optional, List, Iterable, Union
from typing import Awaitable, Collection, Iterable
from copy import copy
import threading

import traceback
import sys

import cython 
cimport cython


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

class Task:

    def __init__(self, taskspace=None, idx=None, state=TaskCreated(), scheduler=None, name=None):
        self.id = id(self)

        
        self.taskspace = taskspace
        self.idx = idx

        self.state = state
        self.scheduler = scheduler

        if isinstance(self.taskspace, TaskSpace):
            self.name = self.unpack_name()
        elif name is None:
            self.name = "UnnamedTask_"+str(idx)
        else:
            #Allow user to specify a name (used for testing and debugging)
            self.name = name

        self.inner_task = core.PyInnerTask(self.id, self)
        self.update_name()

    def unpack_name(self):

        if self.taskspace is not None:
            space_name = self.taskspace._name
        else:
            return self.name

        if isinstance(self.idx, Iterable):
            task_name = "_".join(str(i) for i in (space_name, *self.idx))
        else:
            task_name = "_".join(str(i) for i in (space_name, self.idx))

        return task_name

    def update_name(self):
        name = self.unpack_name()
        self.name = name

        name = name.encode('utf-8')
        self.inner_task.update_name(name)
        

    def instantiate(self, dependencies=None, constraints=None, priority=None):

        self.dependencies = dependencies
        self.constraints = constraints

        self.add_constraints(constraints)
        spawnable_flag = self.add_dependencies(dependencies)

        return spawnable_flag

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

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.inner_task.set_scheduler(scheduler.inner_scheduler)

    def set_state(self, state):
        self.inner_task.set_state(state)

    def get_state(self):
        return self.inner_task.get_state()

    def set_complete(self):
        self.inner_task.set_complete()

    def __repr__(self):
        return f"Task({self.name})"

    def __hash__(self):
            return hash(self.name)

    def __await__(self):
        return (yield TaskAwaitTasks([self], self))

    def add_constraints(self, constraints):
        self.inner_task.add_constraints(constraints)


class ComputeTask(Task):

    def __init__(self, taskspace=None, idx=None, state=TaskCreated(), scheduler=None, name=None):
        super().__init__(taskspace, idx, state, scheduler, name)

    def instantiate(self, function, args, dependencies=None, constraints=None, dataflow=None, priority=0):
        #Holds the original function
        self.base_function = function

        #Holds the function that will be executed (and its continuation)
        self.func = function

        #Holds the arguments to the function
        self.args = args

        #Holds the dataflow object (in/out parrays)
        self.dataflow = dataflow
        
        spawnable_flag = super().instantiate(dependencies, constraints, priority)
        return spawnable_flag

    def _execute_task(self):
        return self.func(self, *self.args)

    def cleanup(self):
        self.func = None
        self.args = None
        self.dataflow = None

    def _finish(self, context):
        pass


#TODO: Data Movement Task  


cpdef flatten_tasks(tasks, list output=[]):
    if isinstance(tasks, Iterable):
        for task in tasks:
            flatten_tasks(task, output)
    else:
        if isinstance(tasks, Task):
            output.append(tasks)
        else:
            raise TypeError("TaskCollection can only contain Tasks")

cdef step(tuple prefix, v):
    return prefix + (v,)

@cython.boundscheck(False)
cpdef cy_parse_index(tuple prefix, index, list index_list, int depth=0, shape=None, start=None):
    #Proof of concept for boundable index parsing
    #TODO: Performance improvements (avoid recursion, etc.)
    shape_flag = (shape is not None)

    cdef int max_dim = len(shape) if shape_flag else 0
    cdef int dim = len(prefix)

    if dim >= max_dim:
        shape = None

    shape_flag = (shape is not None)
    start_flag = (start is not None)

    cdef int lower_boundary = start[dim] if start_flag else 0
    cdef int upper_boundary = lower_boundary + shape[dim] if shape_flag else -1

    cdef int istart = 0
    cdef int istop = 0
    cdef int istep = 1

    if len(index) > 0:
        i, *remainder = index

        if isinstance(i, slice):
            istart = max(i.start, lower_boundary) if i.start is not None else lower_boundary
            if upper_boundary >= 0:
                istop = min(i.stop, upper_boundary) if i.stop is not None else upper_boundary
            else:
                istop = i.stop if i.stop is not None else -1
            istep = i.step or 1

            for v in range(istart, istop, istep):
                cy_parse_index(step(prefix, v), remainder, index_list, depth+1, shape, start)
        elif isinstance(i, Iterable):
            if isinstance(i, str):
                cy_parse_index(step(prefix, i), remainder, index_list, depth+1, shape, start)
            else:
                for v in i:
                    cy_parse_index(step(prefix, v), remainder, index_list, depth+1, shape, start)
        elif isinstance(i, int) or isinstance(i, float):
            if (lower_boundary <= i) and ( (upper_boundary < 0) or (i < upper_boundary) ):
                cy_parse_index(step(prefix, i), remainder, index_list, depth+1, shape, start)
        else:
            cy_parse_index(step(prefix, i), remainder, index_list, depth+1, shape, start)
    else:
        index_list.append(prefix)

def parse_index(prefix, index,  step,  stop):
    """Traverse :param:`index`, update :param:`prefix` by applying :param:`step`, :param:`stop` at leaf calls.

    :param prefix: the initial state
    :param index: the index tuple containing subindexes
    :param step: a function with 2 input arguments (current_state, subindex) which returns the next state, applied for each subindex.
    :param stop: a function with 1 input argument (final_state), applied each time subindexes exhaust.
    """
    if len(index) > 0:
        i, *rest = index
        if isinstance(i, slice):
            for v in range(i.start or 0, i.stop, i.step or 1):
                parse_index(step(prefix, v), rest, step, stop)
        elif isinstance(i, Iterable):
            for v in i:
                parse_index(step(prefix, v), rest, step, stop)
        else:
            parse_index(step(prefix, i), rest, step, stop)
    else:
        stop(prefix)


cpdef get_or_create_tasks(taskspace, list index_list, create=True):
    cdef list task_list = []
    tasks = taskspace._tasks

    for index in index_list:
        if index in tasks:
            task = tasks[index]
            task_list.append(task)
        elif create:
            task = ComputeTask(taskspace=taskspace, idx=index)
            tasks[index] = task
            task_list.append(task)

    return task_list
class TaskCollection:

    def __init__(self, tasks, name=None, flatten=True):
        self._name = name
        if flatten:
            self._tasks = []
            flatten_tasks(tasks, self._tasks)
        else:
            self._tasks = tasks

    @property
    def tasks(self):
        return self._tasks

    def __await__(self):
        return (yield TaskAwaitTasks(self.tasks, None))

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __contains__(self, task):
        return task in self.tasks

    def __repr__(self):
        return "TaskCollection: {}".format(self.tasks)

_task_space_globals = {}

class TaskSpace(TaskCollection):

    def __init__(self, name="", create=True, shape=None, start=None):
        self._name = name
        self._id = id(self)
        self._tasks = {}
        self._create = create

        if shape is not None and not isinstance(shape, tuple):
            shape = (shape,)
        if start is not None and not isinstance(start, tuple):
            start = (start,)

        self.shape = shape
        self.start = start

        global _task_space_globals
        _task_space_globals[self._id] = self

        self._view = None

    def __getitem__(self, index):

        create = self._create
        tasks = self._tasks

        if isinstance(index, int):
            start_flag = (self.start is not None)
            shape_flag = (self.shape is not None)
            lower_boundary = self.start[0] if start_flag else 0
            upper_boundary = lower_boundary + self.shape[0] if shape_flag else -1

            idx = [(index,)] if (index >= lower_boundary) and ((index <= upper_boundary) or (upper_boundary  < 0)) else []
            task_list = get_or_create_tasks(self, idx, create=create)

            if len(task_list) == 1:
                return task_list[0]
            return task_list

        if isinstance(index, str):
            task_list = get_or_create_tasks(self, [(index,)], create=create)
            if len(task_list) == 1:
                return task_list[0]
            return task_list


        if not isinstance(index, tuple):
            index = (index,)

        index_list = []
        cy_parse_index((), index, index_list, shape=self.shape, start=self.start)
        task_list = get_or_create_tasks(self, index_list, create=self._create)

        if len(task_list) == 1:
            return task_list[0]
        else:
            return task_list

    @property
    def tasks(self):
        return self._tasks.values()

    @property
    def name(self):
        return self._name

    @property
    def view(self):
        if self._view is None:
            self._view = TaskSpace(name=self._name, create=False, shape=self.shape, start=self.start)
            self._view._tasks = self._tasks
        return self._view

    def __repr__(self):
        return f"TaskSpace({self._name}, ntasks={len(self)})"