
"""!
@file tasks.pyx
@brief Contains the Task and TaskEnvironment classes, which are used to represent tasks and their execution environments.
"""

from collections import namedtuple, defaultdict
import functools 

from parla.utility.threads import Propagate

cimport core
from parla.cython import core
from parla.cython import device

from parla.common.globals import _Locals as Locals
from parla.common.globals import get_stream_pool, get_scheduler
from parla.common.globals import DeviceType as PyDeviceType
from parla.common.globals import AccessMode, Storage

from parla.cython.cyparray import CyPArray
from parla.common.parray.core import PArray
from parla.common.globals import SynchronizationType as SyncType 
from parla.common.globals import _global_data_tasks


PyDevice = device.PyDevice
PyCUDADevice = device.PyCUDADevice
PyCPUDevice = device.PyCPUDevice
PyArchitecture = device.PyArchitecture
PyCUDAArchitecture = device.PyCUDAArchitecture

DeviceType = PyDeviceType

from abc import abstractmethod, ABCMeta
from typing import Optional, List, Iterable, Union
from typing import Awaitable, Collection, Iterable, FrozenSet
from copy import copy
import threading

import traceback
import sys

import cython 
cimport cython

from parla.cython import device, device_manager

DeviceResourceRequirement = device.DeviceResourceRequirement 
cpu = device_manager.cpu


class TaskState(object, metaclass=ABCMeta):
    """!
    @brief Abstract base class for Task State.
    """

    __slots__ = []

    @property
    @abstractmethod
    def value(self) -> int:
        raise NotImplementedError()
        
    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__


class TaskCreated(TaskState):
    """!
    @brief This state specifies that a task has been created but not yet spawned.
    """

    @property
    def value(self):
        return 0

    @property
    def is_terminal(self):
        return False


class TaskSpawned(TaskState):
    """!
    @brief This state specifies that a task is ready to be mapped to a specific device set
    """
    @property
    def value(self):
        return 1

    @property
    def is_terminal(self):
        return False


class TaskMapped(TaskState):
    """!
    @brief This state specifies that a task has been mapped to a device set, but not yet resered its resources there
    """
    @property
    def value(self):
        return 2

    @property
    def is_terminal(self):
        return False


class TaskReserved(TaskState):
    """!
    @brief This state specifies that a task has reserved its persistent resources (e.g. memory) on its device set. Data movement tasks have been created
    """
    @property
    def value(self):
        return 3

    @property
    def is_terminal(self):
        return False


class TaskReady(TaskState):
    """!
    @brief This state specifies that a task is "ready" to be launched. Its dependencies have been dispatched to hardware queues (or have completed)
    """
    @property
    def value(self):
        return 4

    @property
    def is_terminal(self):
        return False


class TaskRunning(TaskState):
    """!
    @brief This state specifies that a task is executing in a stream.
    """

    __slots__ = ["func", "args", "dependencies"]

    @property
    def value(self):
        return 5

    @property
    def is_terminal(self):
        return False

    # The argument dependencies intentially has no type hint.
    # Callers can pass None if they want to pass empty dependencies.
    def __init__(self, func, args, dependencies: Optional[Iterable] = None):
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


class TaskRunahead(TaskState):
    """!
    @brief State: A task is executing in a stream but the body has completed.
    """
    __slots__ = ["return_value"]

    def __init__(self, ret):
        self.return_value = ret

    @property
    def value(self):
        return 6

    @property
    def is_terminal(self):
        return False

    def __repr__(self):
        return "TaskRunahead({})".format(self.return_value)

class TaskCompleted(TaskState):
    """!
    @brief This state specifies that a task has completed execution.
    """

    __slots__ = ["return_value"]

    @property
    def value(self):
        return 7

    def __init__(self, ret):
        self.return_value = ret

    @property
    def is_terminal(self):
        return True

    def __repr__(self):
        return "TaskCompleted({})".format(self.return_value)


class TaskException(TaskState):
    """!
    @brief This state specifies that a task has completed execution with an exception.
    """

    __slots__ = ["exception", "traceback"]

    @property
    def value(self):
        return 8

    @property
    def is_terminal(self):
        return True

    def __init__(self, exc=None, tb=None):
        self.exception = exc
        self.traceback = tb

    def __repr__(self):
        return "TaskException({})".format(self.exception)


TaskAwaitTasks = namedtuple("AwaitTasks", ["dependencies", "value_task"])


#TODO: Deprecate Task Locals
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
    """!
    @brief Python Task interface. This class is used to represent a task in the task graph.

    A task is a unit of work that can be executed asynchronously. Tasks are created by calling the spawn decorator on a python code block.
    Tasks are scheduled for execution as soon as they are created.

    The task class is a wrapper around a C++ task object. The C++ task object is created when the task is spawned and is destroyed when all references to the task are gone.
    The python interface stores the Python function task body and passes all metadata (mapping and precedence constraints) to the C++ runtime on creations.
    """

    def __init__(self, taskspace=None, idx=None, state=TaskCreated(), scheduler=None, name=None):
        """!
        @brief Create a new task empty object. Task objects are always created empty on first reference and are populated by the runtime when they are spawned. 
        """

        self.id = id(self)

        #TODO(wlr): Should this be a stack for continuation tasks? (so the task has a memory of where it executed from)
        self._environment = None

        self.taskspace = taskspace
        self.idx = idx
        self.func = None
        self.args = None


        self.state = state
        self.scheduler = scheduler

        self.runahead = SyncType.BLOCKING

        if isinstance(self.taskspace, TaskSpace):
            self.name = self.unpack_name()
        elif name is None:
            self.name = "UnnamedTask_"+str(idx)
        else:
            #Allow user to specify a name (used for testing and debugging)
            self.name = name

        self.inner_task = core.PyInnerTask(self.id, self)
        self.update_name()
        self._env = None

    @property
    def env(self):
        """!
        @brief The active TaskEnvironment of the task.
        """
        return self._env

    @env.setter
    def env(self, v):
        self._env = v

    def unpack_name(self) -> str:
        """!
        @brief Create the name of the task from the taskspace and index.
        @return The name of the task.
        """

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
        """!
        @brief Update the name of the task from the taskspace and index.
        """
        name = self.unpack_name()
        self.name = name

        name = name.encode('utf-8')
        self.inner_task.update_name(name)

    def get_name(self) -> str:
        """!
        @brief Get the name of the task.
        @return The name of the task.
        """
        return self.name

    @property
    def environment(self):
        """!
        @brief The active TaskEnvironment of the task.
        """
        return self._environment

    @environment.setter
    def environment(self, env):
        self._environment = env

    def handle_runahead_dependencies(self):
        """!
        @brief Wait (or synchronize) on all events that the task depends on.

        This handles the synchronization through the C++ interface.
        """

        if self.runahead == SyncType.NONE:
            return

        sync_type = self.runahead

        env = self.environment

        if env.has(DeviceType.CPU):
            sync_type = SyncType.BLOCKING

        self.inner_task.handle_runahead_dependencies(int(sync_type))
        

    def py_handle_runahead_dependencies(self):
        """!
        @brief Wait (or synchronize) on all events that the task depends on.

        This handles the synchronization through the Python interface.
        """
        #print("Handling synchronization for task {}".format(self.name), self.runahead, flush=True)
        assert(self.environment is not None)

        if self.runahead == SyncType.NONE:
            return
        elif self.runahead == SyncType.BLOCKING or isinstance(self.env, CPUEnvironment):
            sync_events = self.environment.synchronize_events
        elif self.runahead == SyncType.NON_BLOCKING:
            sync_events = self.environment.wait_events
        else:
            raise NotImplementedError("Unknown synchronization type: {}".format(self.runahead))

        #print("Trying to get dependencies: ", self.name)

        dependencies = self.get_dependencies()

        #print("Dependencies: {}".format(dependencies), flush=True)

        for task in dependencies:
            assert(isinstance(task, Task))

            task_env = task.environment
            assert(task_env is not None)
            if isinstance(task_env, CPUEnvironment):
                continue

            sync_events(task_env)

    def instantiate(self, dependencies=None, list_of_dev_reqs=[], priority=None, dataflow=None, runahead=SyncType.BLOCKING):
        """!
        @brief Add metadata to a blank task object. Includes dependencies, device requirements, priority, and dataflow.
        @param dependencies A list of tasks that this task depends on.
        @param list_of_dev_reqs A list of device requirements/constraints for this task.
        @param priority The priority of the task.
        @param dataflow The collection of CrossPy objects and dependence direction (IN/OUT/INOUT).
        @param runahead The runahead synchronization type of the task. Defaults to SyncType.BLOCKING.
        """

        self.dependencies = dependencies
        self.priority = priority
        self.runahead = runahead

        self.add_dependencies(dependencies)

        # A base task class holds a dataflow since both task types,
        # compute and data move, need it temporarily (e.g., compute tasks
        # need dataflow to create data move tasks) or
        # permanently (e.g., data move tasks need dataflow during its lifecycle).
        # Each data move task only needs a single Parray at this moment,
        # but moving multiple PArrays was also considered as the future work.
        self.add_dataflow(dataflow)

    def _wait_for_dependency_events(self, enviornment):
        pass

    @property
    def result(self):
        """!
        @brief The return value of the task body. This is only valid after the task has completed.

        @return The return value of the task body or an exception if the task threw an exception. Returns None if the task has not completed.
        """

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
        """!
        @brief Run the task body.
        """

        #assert self.assigned, "Task was not assigned to a device before execution"
        #assert isinstance(self.req, EnvironmentRequirements), "Task was not assigned to a enviornment before execution"

        task_state = None
        self.state = TaskRunning(self.func, self.args)
        try:
            task_state = self._execute_task()

            task_state = task_state or TaskRunahead(None)

        except Exception as e:
            tb = traceback.format_exc()
            task_state = TaskException(e, tb)
            self.state = task_state

            print("Exception in Task ", self, ": ", e, tb, flush=True)

            if isinstance(e, KeyboardInterrupt):
                print("You pressed Ctrl+C! In a Task!", flush=True)
                raise e
            #print("Task {} failed with exception: {} \n {}".format(self.name, e, tb), flush=True)

        finally:
            assert(task_state is not None)
            self.state = task_state

    def __await__(self):
        return (yield TaskAwaitTasks([self], self))

    def add_dependencies(self, dependency_list, process=False):
        return self.inner_task.add_dependencies(dependency_list, process)

    def get_num_dependencies(self):
        return self.inner_task.get_num_dependencies()

    def get_num_dependents(self):
        return self.inner_task.get_num_dependents()

    def get_num_blocking_dependencies(self):
        return self.inner_task.get_num_blocking_dependencies()

    def get_num_unmapped_dependencies(self):
        return self.inner_task.get_num_unmapped_dependencies()

    def get_dependencies(self):
        dependency_list = self.inner_task.get_dependencies()
        return dependency_list

    def get_dependents(self):
        dependent_list = self.inner_task.get_dependents()
        return dependent_list

    def get_assigned_devices(self):
        return self.inner_task.get_assigned_devices()

    def create_parray(self, cy_parray: CyPArray, parray_dev_id: int):
        return self.inner_task.create_parray(cy_parray, parray_dev_id)

    def add_dataflow(self, dataflow):
        if dataflow is not None:
            for in_parray_tpl in dataflow.input:
                in_parray = in_parray_tpl[0]
                in_parray_devid = in_parray_tpl[1]
                cy_parray = in_parray.cy_parray
                self.inner_task.add_parray(cy_parray,
                    AccessMode.IN, in_parray_devid)
            for out_parray_tpl in dataflow.output:
                out_parray = out_parray_tpl[0]
                out_parray_devid = out_parray_tpl[1]
                cy_parray = out_parray.cy_parray
                self.inner_task.add_parray(cy_parray,
                    AccessMode.OUT, out_parray_devid)
            for inout_parray_tpl in dataflow.inout:
                inout_parray = inout_parray_tpl[0]
                inout_parray_devid = inout_parray_tpl[1]
                cy_parray = inout_parray.cy_parray
                self.inner_task.add_parray(cy_parray,
                    AccessMode.INOUT, inout_parray_devid)

    def notify_dependents_wrapper(self):
        """!
        @brief Mock dependents interface only used for testing. Notify dependents should be called internall by the scheduler
        """
        status = self.inner_task.notify_dependents_wrapper()
        return status

    def set_scheduler(self, scheduler):
        """!
        @brief Set the scheduler the task has been spawned by.
        """
        self.scheduler = scheduler
        self.inner_task.set_scheduler(scheduler.inner_scheduler)

    def set_state(self, state):
        """!
        @brief Set the state of the task (passed to the C++ runtime)
        """
        self.inner_task.set_state(state)

    def get_state(self):
        """!
        @brief Get the state of the task (from the C++ runtime)
        """
        return self.inner_task.get_state()

    def set_complete(self):
        self.inner_task.set_complete()

    def set_device_reqs(self, device_reqs):

        """!
        @brief Set the device requirements of the task.
        @param device_reqs A list of device requirements. Each device requirement can be a single device, a single architecture, or a tuple of devices and architectures.
        """

        # device_reqs: a list of device requirements,
        # a list of list of devices and frozensets
        # a list of a single frozenset
        for req in device_reqs:
            if isinstance(req, DeviceResourceRequirement):
                # Single device.
                self.inner_task.add_device_req(
                    req.device.get_cy_device(),
                    req.res_req.memory, req.res_req.vcus)
            elif isinstance(req, FrozenSet):
                # Single architecture
                self.inner_task.begin_arch_req_addition()
                for member in req:
                    self.inner_task.add_device_req(
                        member.device.get_cy_device(),
                        member.res_req.memory, member.res_req.vcus)
                self.inner_task.end_arch_req_addition()
            elif isinstance(req, List):
                # Multi-optional requirements
                self.inner_task.begin_multidev_req_addition()
                for member in req: 
                    self.set_device_reqs([member])
                self.inner_task.end_multidev_req_addition()

    def __repr__(self):
        return f"Task({self.name})"

    def __hash__(self):
            return hash(self.name)

    def __await__(self):
        return (yield TaskAwaitTasks([self], self))

    def add_stream(self, stream):
        """
        @brief Record a python managed cupy stream to the task.
        """
        self.inner_task.add_stream(stream)

    def add_event(self, event):
        """
        @brief Record a python managed cupy event to the task.
        """
        self.inner_task.add_event(event)

    def cleanup(self):
        raise NotImplementedError()


class ComputeTask(Task):
    """!
    @brief A compute task is a task that executes a user defined Python function on a device.
    """


    def __init__(self, taskspace=None, idx=None, state=TaskCreated(), scheduler=None, name=None):
        super().__init__(taskspace, idx, state, scheduler, name)

    def instantiate(self, function, args, dependencies=None, dataflow=None, priority=0, runahead=SyncType.BLOCKING):
        """!
        @brief Instantiate the task with a function and arguments.
        @param function The function to execute.
        @param args The arguments to the function.
        @param dependencies A list of tasks that this task depends on.
        @param dataflow The dataflow object that describes the data dependencies of the task. (Crosspy and data direction (IN/OUT/INOUT))
        @param priority The priority of the task.
        @param runahead The type of synchronization the task uses for runahead scheduling.
        """

        #Holds the original function
        self.base_function = function

        #Holds the function that will be executed (and its continuation)
        self.func = function

        #Holds the arguments to the function
        self.args = args

        #Holds the dataflow object (in/out parrays)
        self.dataflow = dataflow
        
        super().instantiate(dependencies=dependencies, priority=priority, dataflow=dataflow, runahead=runahead)

    def _execute_task(self):
        """!
        @brief Run the task body with the saved arguments. If the body is a continuation, run the continuation.
        """
        return self.func(self, *self.args)

    def cleanup(self):
        """!
        @brief Cleanup the task by removing the function, arguments, and references to its data objects.
        """
        self.func = None
        self.args = None
        self.dataflow = None

    def _finish(self, context):
        pass


class DataMovementTask(Task):
    """!
    @brief A data movement task is a task that moves data between devices. It is not user defined.
    """

    def __init__(self, parray: PArray=None, access_mode=None, \
        assigned_devices: List[PyDevice]=None, taskspace=None, \
        idx=0, state=TaskCreated(), scheduler=None, name=None):
        super().__init__(taskspace, idx, state, scheduler, name)
        self.parray = parray

        self.access_mode = access_mode
        self.assigned_devices = assigned_devices

    def instantiate(self, attrs: core.DataMovementTaskAttributes, scheduler, runahead=SyncType.BLOCKING):
        """!
        @brief Instantiate the data movement task with attributes from the C++ runtime.
        @param attrs The attributes of the data movement task.
        @param scheduler The scheduler that the task is created under.
        @param runahead The type of synchronization the task uses for runahead scheduling.
        """
        self.name = attrs.name
        self.parray = attrs.parray
        self.access_mode = attrs.access_mode
        self.assigned_devices = attrs.assigned_devices
        self.scheduler = scheduler
        self.inner_task.set_c_task(attrs.c_attrs)
        self.inner_task.set_py_task(self)
        self.dev_id = attrs.dev_id
        self.runahead = runahead
        self.dependencies = self.get_dependencies()

    def _execute_task(self):
        """!
        @brief Run the data movement task. Calls the PArray interface to move the data to the assigned devices.
        Devices are given by the local relative device id within the TaskEnvironment.
        """
        write_flag = True if self.access_mode != AccessMode.IN else False

        #TODO: Get device manager from task environment instead of scheduler at creation time
        device_manager = self.scheduler.device_manager
        """
        for device in self.assigned_devices:
            global_device_id = device.get_global_id()
            self.parray._auto_move(device_manager.get_parray_id(global_device_id),
                                   write_flag)
        """
        target_dev = self.assigned_devices[0]
        global_id = target_dev.get_global_id()
        parray_id = device_manager.globalid_to_parrayid(global_id)
        self.parray._auto_move(parray_id, write_flag)
        #print(self, "Move PArray ", self.parray.ID, " to a device ", parray_id, flush=True)
        #print(self, "STATUS: ", self.parray.print_overview())
        return TaskRunahead(0)

    def cleanup(self):
        self.parray = None

######
# Task Environment
######

def create_device_env(device):
    """!
    @brief Create a terminal device environment from a PyDevice.
    @param device The PyDevice to create the environment from.
    """
    if isinstance(device, PyCPUDevice):
        return CPUEnvironment(device), DeviceType.CPU
    elif isinstance(device, PyCUDADevice):
        return GPUEnvironment(device), DeviceType.CUDA

def create_env(sources):
    """!
    @brief Create the union  TaskEnvironment from a list of TaskEnvironments or PyDevices.
    @param sources The list of PyDevices (or Environments) to create the new environment from.
    """

    targets = []

    for env in sources:
        if isinstance(env, PyDevice):
            device = env
            new_env, dev_type = create_device_env(env)
            targets.append(new_env)

    if len(targets) == 1:
        return targets[0]
    else:
        return TaskEnvironment(targets)

class TaskEnvironment:

    """!
    @brief A TaskEnvironment is a collection of devices or other TaskEnvironments used to coordinate and synchronize kernels in the Task body.
    """

    def __init__(self, environment_list, blocking=False):

        self._store = Storage()

        self.device_dict = defaultdict(list)

        self.env_list = []
        self.stream_list = []
        self.is_terminal = False
        self.blocking = blocking
        self._device = None

        self.device_list = []
        self._global_device_ids = set()
        self.event_dict = {}
        self.event_dict['default'] = None

        for env in environment_list:
            for dev in env.devices:
                self.device_list.append(dev)
                self.device_dict[dev.architecture].append(dev)

            self.stream_list.extend(env.streams)
            self._global_device_ids  = self._global_device_ids.union(env.global_ids)
            self.env_list.append(env)

    @property
    def global_ids(self):
        """
        Return the global Parla device ids of the devices in this environment.
        """
        return self._global_device_ids

    @property
    def gpu_ids(self):
        """
        Returns the CUDA_VISIBLE_DEVICES ids of the GPU devices in this environment.
        """
        return [device_env.get_parla_device().id for device_env in self.device_dict[DeviceType.CUDA]]

    @property
    def gpu_id(self):
        """
        Returns the CUDA_VISIBLE_DEVICES id of the first GPU device in this environment.
        """
        return self.device_dict[DeviceType.CUDA][0].get_parla_device().id

    def __repr__(self):
        return f"TaskEnvironment({self.env_list})"

    def get_parla_device(self):
        """
        Returns the Parla device associated with this environment.
        """
        return self._device

    def get_library_device(self):
        """
        Returns the library device associated with this environment. (e.g. cupy.cuda.Device)
        """
        return self._device.device

    def has(self, device_type):
        """
        Returns True if this environment has a device of the given type.
        """
        return device_type in self.device_dict

    @property
    def streams(self):
        return self.stream_list

    @property
    def stream(self):
        """
        Returns the Parla stream associated with this environment.
        """
        return self.streams[0]

    @property
    def cupy_stream(self):
        """
        Returns the cupy stream associated with this environment.
        """
        return self.stream.stream

    def loop(self, envlist=None):
        if envlist is None:
            envlist = self.contexts
        
        for env in envlist:
            env.__enter__()
            yield env
            env.__exit__(None, None, None)

    def get_devices(self, arch):
        return self.device_dict[arch]

    def get_all_devices(self):
        #return sum(self.device_dict.values(), [])
        return self.device_list

    @property
    def contexts(self):
        return self.env_list

    @property
    def devices(self):
        #TODO: Improve this
        devices = self.get_all_devices()
        #print(f"Devices: {devices}")
        return devices
    
    @property
    def device(self):
        return self.devices[0]

    def get_cupy_devices(self):
        return [dev.device for dev in self.get_devices(DeviceType.CUDA)]

    def synchronize(self, events=False, tags=['default'], return_to_pool=True):
        #print(f"Synchronizing {self}..", flush=True)

        if self.is_terminal:
            if events:
                for tag in tags:
                    #print("SELF: ", self, f"Synchronizing on event {tag}..", flush=True)
                    self.synchronize_event(tag=tag)
            else:
                for stream in self.stream_list:
                    #print("SELF: ", self, f"Synchronizong on stream {stream}", flush=True)
                    stream.synchronize()

            if return_to_pool:
                stream_pool = get_stream_pool()
                for stream in self.stream_list:
                    stream_pool.return_stream(stream)
        else:
            for env in self.env_list:
                #print("Non terminal: Recursing", flush=True)
                env.synchronize(events=events, tags=tags)

    def __enter__(self):
        #print("Entering environment:", self.env_list, flush=True)

        if len(self.env_list) == 0:
            raise RuntimeError("[TaskEnvironment] No environment or device is available.")

        Locals.push_context(self)
        self.devices[0].enter_without_context()

        return self

    def enter_without_context(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        #print("Exiting environment", self.env_list, flush=True)
        ret = False
        self.devices[0].exit_without_context(exc_type, exc_val, exc_tb)
        Locals.pop_context()
        
        return ret 

    def exit_without_context(self, exc_type, exc_val, exc_tb):
        return False

    def __getitem__(self, index):

        if isinstance(index, int):
            return self.env_list[index]
        
        return create_env(self.env_list[index])


    def store(self, key, value):
        self._store.store(key, value)

    def retrieve(self, key):
        return self._store.retrieve(key)

    @property
    def storage(self):
        return self._store

    def __len__(self):
        return len(self.env_list)

    def return_streams(self):
        for env in self.env_list:
            env.return_streams()
        
        stream_pool = get_stream_pool()
        for stream in self.stream_list:
            stream_pool.return_stream(stream)

    #TODO: MOVE THIS TO C++!!!!
    def finalize(self):
        stream_pool = get_stream_pool()

        for env in self.env_list:
            env.finalize()
        
        for stream in self.stream_list:
            stream.synchronize()
            stream_pool.return_stream(stream)

    def __contains__(self, obj):
        #TODO(wlr): Add optional support for CuPy device 
        if isinstance(obj, PyDevice):
            return obj in self.device_list
        elif isinstance(obj, TaskEnvironment):
            return obj.global_ids.issubset(self.global_ids)
        else:
            raise TypeError("The comparison is not supported. (Supported Types: TaskEnvironment)")

    def parfor(self, envlist=None):

        if envlist is None:
            envlist = self.contexts

        def deco(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                res = [Exception('Parallel Launcher [%s] raised an exception!' % (
                    func.__name__))]

                def EnvHandler(env, idx):
                    Locals.index = idx 
                    env.__enter__()
                    try:
                        res[0] = func(env, *args, **kwargs)
                    except Exception as e:
                        res[0] = e
                    finally:
                        env.__exit__(None, None, None)

                thread_list = []
                return_list = []
                for idx, env in enumerate(envlist):
                    thread_list.append(Propagate(target=EnvHandler, args=(env, idx)))
                try:
                    for t in thread_list:
                        t.start()
                    for t in thread_list:
                        t.join()
                        return_list.append(t.value)
                except Exception as e:
                    print('Unhandled exception in Propagate wrapper', flush=True)
                    raise e

                ret = res[0]
                if isinstance(ret, BaseException):
                    raise ret
                return ret
            return wrapper()
        return deco

    def __contains__(self, obj):
        if isinstance(obj, PyDevice):
            return obj.global_id in self._global_device_ids
        elif isinstance(obj, TaskEnvironment):
            return obj.global_ids.issubset(self._global_device_ids)
        else:
            raise TypeError("Invalid type for __contains__")


    def wait_events(self, env, tags=['default']):
        """
        Wait for tagged events in the given environment on all streams in this environment.
        """

        #print("Waiting for events", env, tags, flush=True)

        if not isinstance(tags, list):
            tags = [tags]

        for device in env.devices:
            for stream in device.streams:
                for tag in tags:
                    #print("++Waiting for event", device, stream, tag, flush=True)
                    device.wait_event(stream=stream, tag=tag)

    def synchronize_events(self, env, tags=['default']):
        """
        Synchronize tagged events in the given environment on all streams in this environment.
        """

        if not isinstance(tags, list):
            tags = [tags]

        for device in env.devices:
            for stream in device.streams:
                for tag in tags:
                    #print("++Synchronizing event", device, stream, tag, flush=True)
                    device.synchronize_event(tag=tag)
    
    def record_events(self, tags=['default']):
        """
        Record tagged events on all streams in this environment.
        """

        if not isinstance(tags, list):
            tags = [tags]

        for device in self.devices:
            for stream in device.streams:
                for tag in tags:
                    #print("--Recording event", device, stream, tag, flush=True)
                    device.record_event(stream=stream, tag=tag)

    def create_events(self, tags=['default']):
        """
        Create tagged events on all devices in this environment.
        """

        if not isinstance(tags, list):
            tags = [tags]

        for device in self.devices:
            for tag in tags:
                device.create_event(tag=tag)

    def write_to_task(self, task):

        for device in self.devices:
            device.write_to_task(task)

    def write_streams_to_task(self, task):
            
        for device in self.devices:
            device.write_streams_to_task(task)
      

class TerminalEnvironment(TaskEnvironment):

    """!
    @brief An endpoint TaskEnvironment representing a single device. These are where most actual computation will take place.

    @details A TerminalEnviornment is an edpoint TaskEnvironment that is made of a single device (CPU or GPU). If they are a GPU they will set the current CuPy context accordingly.
    """

    def  __init__(self,  device, blocking=False):
        super(TerminalEnvironment, self).__init__([], blocking=blocking)
        self.device_dict[device.architecture].append(self)
        self.device_list.append(self)
        self._device = device
        self._arch_type = device.architecture
        self.is_terminal = True

        self._global_device_ids = {device.get_global_id()}

    def __repr__(self):
        return f"TerminalEnvironment({self._device})"

    @property
    def contexts(self):
        return [self]

    @property
    def devices(self):
        return [self]

    @property
    def device(self):
        return self

    @property
    def architecture(self):
        return self._arch_type

    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, PyDevice):
            return self._device == other
        elif isinstance(other, TerminalEnvironment):
            return self._device == other._device
        else:
            return False

    def __hash__(self):
        return hash(self._device)

    def __call__(self):
        return self._device

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index == 0:
            return self
        else:
            raise IndexError("TerminalEnvironment only has one device.")

    def record_event(self, stream=None, tag='default'):
        """!
        @brief Record a CUDA event on the current stream. 
        """

        if stream is None:
            stream = Locals.stream 
        
        if tag not in self.event_dict:
            raise RuntimeError("Event must be created before recording.")

        event = self.event_dict[tag]
        if event is not None:
            #print("TEST RECORD: ", event, stream.stream)
            event.record(stream.stream)

    def synchronize_event(self, tag='default'):
        """!
        @brief Synchronize host thread to the tagged CUDA event (sleep or waiting). 
        """

        if tag not in self.event_dict:
            raise RuntimeError("Event must be created before synchronizing.")

        event = self.event_dict[tag]

        if event is not None:
            #print("TEST EVENT SYNC: ", event, flush=True)
            event.synchronize()

    def wait_event(self, stream=None, tag='default'):
        """!
        @brief Submit a cross-stream wait on the tagged CUDA event to the current stream. All further work submitted on the current stream will wait until the tagged event is recorded.
        """
        if tag not in self.event_dict:
            raise RuntimeError("Event must be created before waiting.")

        event = self.event_dict[tag]

        if event is not None:
            #print("TEST WAIT EVENT: ", stream, event)
            stream.wait_event(event)

    def create_event(self, stream=None, tag='default'):
        """!
        @brief Create a CUDA event on the current stream.  It can be used for synchronization or cross-stream waiting. It is not recorded by default at creation.
        """
        if stream is None:
            stream = Locals.stream
        self.event_dict[tag] = stream.create_event()

    def write_to_task(self, task):
        """!
        @brief Store stream and event pointers in C++ task
        """
        for stream in self.streams:
            task.add_stream(stream.stream)
        
        #for event in self.event_dict.values():
        #    task.add_event(event)

        #Note: only adding default event for now
        task.add_event(self.event_dict['default'])

    def write_streams_to_task(self, task):
        """!
        @brief Record stream pointers into C++ task
        """
        for stream in self.streams:
            task.add_stream(stream.stream)

    def write_events_to_task(self, task):
        """!
        @brief Record event pointers in C++ task
        """
        #for event in self.event_dict.values():
                #    task.add_event(event)
        task.add_event(self.event_dict['default'])
        

        

    
class CPUEnvironment(TerminalEnvironment):

    def __init__(self,  device, blocking=False):
        super(CPUEnvironment, self).__init__(device, blocking=blocking)

    def __repr__(self):
        return f"CPUEnvironment({self._device})"

    def __enter__(self):
        #print("Entering CPU Environment: ", self, flush=True)
        Locals.push_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        #print("Exiting CPU Environment: ", self, flush=True)
        Locals.pop_context()
        return False

    def __len__(self):
            return 1

    def __getitem__(self, index):
        if index == 0:
            return self

    def finalize(self):
        pass

    def return_streams(self):
        pass

class GPUEnvironment(TerminalEnvironment):

    def __init__(self, device, blocking=False):
        super(GPUEnvironment, self).__init__(device, blocking=blocking)

        stream_pool = get_stream_pool()
        stream = stream_pool.get_stream(device=device)
        self.stream_list.append(stream)

        self.event_dict['default'] = stream.create_event()


    def __repr__(self):
        return f"GPUEnvironment({self._device})"


    def __enter__(self):
        #print("Entering GPU Environment: ", self, flush=True)
        Locals.push_context(self)
        self.active_stream = self.stream_list[0]
        ret_stream = self.active_stream.__enter__()
        return self

    def enter_without_context(self):
        self.active_stream = self.stream_list[0]
        ret_stream = self.active_stream.__enter__()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        #print("Exiting GPU Environment: ", self, flush=True)
        ret = False
        self.active_stream.__exit__(exc_type, exc_val, exc_tb)
        Locals.pop_context()
        return ret 

    def exit_without_context(self, exc_type, exc_val, exc_tb):
        ret = False
        self.active_stream.__exit__(exc_type, exc_val, exc_tb)
        return ret

    def finalize(self):
        stream_pool = get_stream_pool()
        for stream in self.stream_list:
            stream.synchronize()
            stream_pool.return_stream(stream)

    def return_streams(self):
        stream_pool = get_stream_pool()
        for stream in self.stream_list:
            stream_pool.return_stream(stream)


#######
# Task Collections
#######

cpdef flatten_tasks(tasks, list output=[]):

    #Unpack any TaskCollections
    if isinstance(tasks, TaskCollection):
        tasks = tasks.tasks

    if isinstance(tasks, list) or isinstance(tasks, tuple):
        for i in range(0, len(tasks)):
            task = tasks[i]
            flatten_tasks(task, output)
    elif isinstance(tasks, Task):
        output.append(tasks)
    elif isinstance(tasks, dict):
        keys = tasks.keys()
        for i in range(0, len(keys)):
            task = tasks[keys[i]]
            flatten_tasks(task, output)
    elif isinstance(tasks, Iterable):
        #NOTE: This is not threadsafe if iterated concurrently
        for task in tasks:
            flatten_tasks(task, output)
    else:
        raise TypeError("TaskCollections can only contain Tasks or Iterable Containers of Tasks")

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

    #TODO(wlr): Iterable check should be more robust (try/catch)

    if len(index) > 0:
        i, *remainder = index

        if isinstance(i, TaskCollection):
            i = i.tasks

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
            elif isinstance(i, tuple) or isinstance(i, list):
                for k in range(0, len(i)):
                    cy_parse_index(step(prefix, i[k]), remainder, index_list, depth+1, shape, start)
            elif isinstance(i, dict):
                keys = i.keys()
                for k in range(0, len(keys)):
                    cy_parse_index(step(prefix, i[keys[k]]), remainder, index_list, depth+1, shape, start)
            else:
                #NOTE: This is not threadsafe if the iterator is shared
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

        if isinstance(i, TaskCollection):
            i = i.tasks 

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
    cdef list new_tasks = []
    cdef list new_index = []

    tasks = taskspace._tasks

    for i in range(0, len(index_list)):
        index = index_list[i]

        if index in tasks:
            task = tasks[index]
            task_list.append(task)
        elif create:
            task = ComputeTask(taskspace=taskspace, idx=index)
            tasks[index] = task
            task_list.append(task)
            new_tasks.append(task)
            new_index.append(index)

    return task_list, (new_tasks, new_index)
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
        return task in self._tasks
        
    def __repr__(self):
        return "TaskCollection: {}".format(self.tasks)

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self._tasks) == id(self._tasks)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return TaskCollection(self._tasks + other._tasks)

    def __iadd__(self, other):
        self._tasks += other._tasks
        return self



class TaskList(TaskCollection):

    def __init__(self, tasks, name=None, flatten=True):

        if isinstance(tasks, TaskList):
            self._name = tasks._name
            self._tasks = tasks.tasks
        else:
            super().__init__(tasks, name, flatten)

    def __getitem__(self, index):
        task_list = self.tasks[index]

        if isinstance(task_list, list):
            return TaskList(task_list, flatten=False)
        else:
            #Return a single task
            return task_list

    def __repr__(self):
        return "TaskList: {}".format(self.tasks)

    def __add__(self, other):
        return TaskList(self._tasks + other._tasks)

    def __iadd__(self, other):
        self._tasks += other._tasks
        return self

cpdef wait(barrier):

    if isinstance(barrier, core.CyTaskList):
        barrier = BackendTaskList(barrier)

    barrier.wait()

class AtomicTaskList(TaskList):

    def __init__(self, tasks, name=None, flatten=True):
        super().__init__(tasks, name, flatten)
        self.inner_barrier = core.PyTaskBarrier(self.tasks)

    def __repr__(self):
        return "AtomicTaskList: {}".format(self.tasks)

    def __add__(self, other):
        return AtomicTaskList(self._tasks + other._tasks)

    def __iadd__(self, other):
        raise TypeError("Cannot modify an AtomicTaskList")

    def wait(self):
        self.inner_barrier.wait()

class BackendTaskList(TaskList):

    def __init__(self, tasks, name=None, flatten=True):
        self.inner_barrier = core.PyTaskBarrier(tasks)
        self._tasks = None
        self._name = name 

    def __repr__(self):
        return "BackendTaskList: {}"

    def wait(self):
        self.inner_barrier.wait()


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
            task_list, _= get_or_create_tasks(self, idx, create=create)

            if len(task_list) == 1:
                return task_list[0]
            return TaskList(task_list)

        if isinstance(index, str):
            task_list, _ = get_or_create_tasks(self, [(index,)], create=create)
            if len(task_list) == 1:
                return task_list[0]
            return TaskList(task_list)


        if not isinstance(index, tuple):
            index = (index,)

        index_list = []
        cy_parse_index((), index, index_list, shape=self.shape, start=self.start)
        task_list, _ = get_or_create_tasks(self, index_list, create=self._create)

        if len(task_list) == 1:
            return task_list[0]
        else:
            return TaskList(task_list)

    @property
    def tasks(self):
        return self._tasks.values()

    @property
    def name(self):
        return self._name

    @property
    def view(self):
        if self._view is None:
            self._view = type(self)(name=self._name, create=False, shape=self.shape, start=self.start)
            self._view._tasks = self._tasks
        return self._view

    def __repr__(self):
        return f"TaskSpace({self._name}, ntasks={len(self)})"

    def __add__(self, other):
        merged_dict = {**self._tasks, **other._tasks}
        merged_name = f"{self._name} + {other._name}"
        new_space = type(self)(name=merged_name, create=False, shape=self.shape, start=self.start)
        new_space._tasks = merged_dict
        return new_space

    def __iadd__(self, other):
        self._tasks.update(other._tasks)
        return self


class AtomicTaskSpace(TaskSpace):

    def __init__(self, name="", create=True, shape=None, start=None):
        super().__init__(name, create, shape, start)
        self.inner_space = core.PyTaskBarrier()

    def __repr__(self):
        return f"AtomicTaskSpace({self._name}, ntasks={len(self)})"

    
    def __getitem__(self, index):

        create = self._create
        tasks = self._tasks

        if isinstance(index, int):
            start_flag = (self.start is not None)
            shape_flag = (self.shape is not None)
            lower_boundary = self.start[0] if start_flag else 0
            upper_boundary = lower_boundary + self.shape[0] if shape_flag else -1

            idx = [(index,)] if (index >= lower_boundary) and ((index <= upper_boundary) or (upper_boundary  < 0)) else []
            task_list, (new_tasks, new_idx) = get_or_create_tasks(self, idx, create=create)

            #self.inner_space.add_tasks(new_idx, new_tasks)
            self.inner_space.add_tasks(new_tasks)

            if len(task_list) == 1:
                return task_list[0]

            return AtomicTaskList(task_list)

        if isinstance(index, str):
            task_list, (new_tasks, new_index) = get_or_create_tasks(self, [(index,)], create=create)
            #self.inner_space.add_tasks(new_idx, new_tasks)
            self.inner_space.add_tasks(new_tasks)

            if len(task_list) == 1:
                return task_list[0]

            return AtomicTaskList(task_list)


        if not isinstance(index, tuple):
            index = (index,)

        index_list = []
        cy_parse_index((), index, index_list, shape=self.shape, start=self.start)
        task_list, (new_tasks, new_index) = get_or_create_tasks(self, index_list, create=self._create)
        #self.inner_space.add_tasks(new_idx, new_tasks)
        self.inner_space.add_tasks(new_tasks)

        if len(task_list) == 1:
            return task_list[0]

        return AtomicTaskList(task_list)

    def wait(self):
        self.inner_space.wait()


#TODO(wlr): This is incredibly experimental. 
class BackendTaskSpace(TaskSpace):

    def __init__(self, name="", create=True, shape=None, start=None):
        super().__init__(name, create, shape, start)
        self.inner_space = core.PyTaskSpace()

    def __repr__(self):
        return f"BackendTaskspace({self._name}, ntasks={len(self)})"

    
    def __getitem__(self, index):

        create = self._create
        tasks = self._tasks

        if isinstance(index, int):
            start_flag = (self.start is not None)
            shape_flag = (self.shape is not None)
            lower_boundary = self.start[0] if start_flag else 0
            upper_boundary = lower_boundary + self.shape[0] if shape_flag else -1

            index_list = [(index,)] if (index >= lower_boundary) and ((index <= upper_boundary) or (upper_boundary  < 0)) else []
            task_list, (new_tasks, new_idx) = get_or_create_tasks(self, index_list, create=create)

            #self.inner_space.add_tasks(new_idx, new_tasks)
            self.inner_space.add_tasks(new_tasks)

            return_list = core.CyTaskList()
            self.inner_space.get_tasks(index_list, return_list)
            return return_list

        if isinstance(index, str):
            index_list = [(index,)]
            task_list, (new_tasks, new_index) = get_or_create_tasks(self, index_list, create=create)
            #self.inner_space.add_tasks(new_idx, new_tasks)
            self.inner_space.add_tasks(new_tasks)

            return_list = core.CyTaskList()
            self.inner_space.get_tasks(index_list, return_list)
            return return_list


        if not isinstance(index, tuple):
            index = (index,)

        index_list = []
        cy_parse_index((), index, index_list, shape=self.shape, start=self.start)
        task_list, (new_tasks, new_index) = get_or_create_tasks(self, index_list, create=self._create)
        #self.inner_space.add_tasks(new_idx, new_tasks)
        self.inner_space.add_tasks(new_tasks)

        return_list = core.CyTaskList()
        self.inner_space.get_tasks(index_list, return_list)
        return return_list


    def wait(self):
        self.inner_space.wait()
