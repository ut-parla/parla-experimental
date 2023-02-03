from abc import abstractmethod, ABCMeta
from typing import Collection, Optional, Union, List, Dict
import threading
from collections import deque, namedtuple, defaultdict
import inspect 

import traceback
import sys

#cimport tasks
from parla.cython import tasks

cimport core
from parla.cython import core

TaskID = tasks.TaskID
Task = tasks.Task
ComputeTask = tasks.ComputeTask

import nvtx

PyInnerScheduler = core.PyInnerScheduler
PyInnerWorker = core.PyInnerWorker
PyInnerTask = core.PyInnerTask

class TaskBodyException(RuntimeError):
    pass

class SchedulerException(RuntimeError):
    pass

class WorkerThreadException(RuntimeError):
    pass


#TODO: Unspawned Dependencies need to be rethought and implemented at the C++ level

class _SchedulerLocals(threading.local):
    def __init__(self):
        super(_SchedulerLocals, self).__init__()
        self._scheduler_context_stack = []

    @property
    def scheduler_context(self):
        if self._scheduler_context_stack:
            return self._scheduler_context_stack[-1]
        else:
            raise Exception("No scheduler context")

_scheduler_locals = _SchedulerLocals()

def get_scheduler_context():
    return _scheduler_locals.scheduler_context

class SchedulerContext:

    #TODO: Add enviornments back

    @property
    @abstractmethod
    def scheduler(self) -> "Scheduler":
        raise NotImplementedError()

    def __enter__(self):
        _scheduler_locals._scheduler_context_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _scheduler_locals._scheduler_context_stack.pop()

class ControllableThread(threading.Thread):

    def __init__(self):
        super().__init__()
        self._should_run = True

    def stop(self):
        #print("Stopping Thread:", self, flush=True)
        with self._monitor:
            self._should_run = False
            self._monitor.notify_all()

    @abstractmethod
    def run(self):
        pass


class WorkerThread(ControllableThread, SchedulerContext):
    def __init__(self, scheduler, index):
        super().__init__()
        self._scheduler = scheduler
        self._monitor = threading.Condition(threading.Lock())

        self.index = index

        self.task = None
        self.status = "Initializing"

        self.inner_worker = PyInnerWorker(self)

        #Add the worker to the scheduler pool of all workers (not yet active)
        scheduler.inner_scheduler.add_worker(self.inner_worker)

    @property
    def scheduler(self):
        return self._scheduler

    def assign_task(self, task):
        #print("Worker waiting to assign task", flush=True)
        with self._monitor:
            if self.task:
                raise Exception("Worker already has a task")
            self.task = task
            #print("Worker assigned task. Waking thread...", flush=True)
            self._monitor.notify()

    def remove_task(self):
        with self._monitor:
            if not self.task:
                raise Exception("Worker does not have a task")
            self.task = None

    def run(self):
        try:
            #A worker thread is a scheduler context
            with self:
                #TODO: Perform any thread initialization on enviornment components

                #Add the worker to the scheduler pool of active & availabe workers
                self.scheduler.inner_scheduler.enqueue_worker(self.inner_worker)
                with self.scheduler.start_monitor:
                    #print("NOTIFYING", flush=True)
                    self.scheduler.start_monitor.notify_all()

                while self._should_run:
                    self.status = "Waiting"
                    #print("WAITING", flush=True)

                    #with self._monitor:
                    #    if not self.task:
                    #        self._monitor.wait()
                    nvtx.push_range(message="worker::wait", domain="Python Runtime", color="blue")
                    self.inner_worker.wait_for_task()

                    self.task = self.inner_worker.get_task()
                    nvtx.pop_range(domain="Python Runtime")

                    #print("THREAD AWAKE", self.index, self.task, self._should_run, flush=True)

                    self.status = "Running"

                    if isinstance(self.task, ComputeTask):
                        active_task = self.task 

                        #print("Running Task", self.index, active_task.taskid.full_name, flush=True)
                        nvtx.push_range(message="worker::run", domain="Python Runtime", color="blue")
                        active_task.run()
                        nvtx.pop_range(domain="Python Runtime")
                        #print("Finished Task", self.index, active_task.taskid.full_name, flush=True)

                        nvtx.push_range(message="worker::cleanup", domain="Python Runtime", color="blue")
                        #TODO: Add better exception handling
                        if isinstance(active_task.state, tasks.TaskException):
                            raise TaskBodyException(active_task.state.exception)

                        if isinstance(active_task.state, tasks.TaskRunning):
                            #print("CONTINUATION: ", active_task.taskid.full_name, active_task.state.dependencies, flush=True)
                            active_task.dependencies = active_task.state.dependencies
                            active_task.func = active_task.state.func
                            active_task.args = active_task.state.args

                            active_task.inner_task.clear_dependencies()
                            active_task.add_dependencies(active_task.dependencies, process=False)

                        self.inner_worker.remove_task()
                        self.scheduler.inner_scheduler.task_cleanup(self.inner_worker, active_task.inner_task, active_task.state.value)
                        nvtx.pop_range(domain="Python Runtime")

                    elif self._should_run:
                        raise WorkerThreadException("%r Worker: Woke without a task", self.index)
                    else:
                        #print("Worker Thread Stopping", flush=True)
                        break

        except Exception as e:
            print("Exception in Worker Thread", e, flush=True)
            self.scheduler.exception_stack.append(e)
            self.scheduler.stop()

            if isinstance(e, TaskBodyException):
                raise WorkerThreadException("Unhandled Exception in Task") from e
            else:
                raise WorkerThreadException("Unhandled Exception on "+str(self))

    def stop(self):
        super().stop()
        self.inner_worker.stop()
        #print("Stopped Thread", self, flush=True)

class Scheduler(ControllableThread, SchedulerContext):

    def __init__(self, n_threads=6, period=0.001):
        super().__init__()

        self.start_monitor = threading.Condition(threading.Lock())

        self._monitor = threading.Condition(threading.Lock())

        self.exception_stack = []

        #TODO: Handle resources better
        resources = 1.0

        self.inner_scheduler = PyInnerScheduler(n_threads, resources, self)

        self.worker_threads = [WorkerThread(self, i) for i in range(n_threads)]

        with self.start_monitor:
            for thread in self.worker_threads:
                thread.start()
            #print("Scheduler: Waiting at least one thread to Spawn", flush=True)
            self.start_monitor.wait()

        self.start()

    @property
    def scheduler(self):
        return self

    def __enter__(self):
        if self.inner_scheduler.get_num_active_tasks() != 1:
            raise SchedulerException("Schedulers can only have a single scope.")
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        #print("Scheduler: Exiting", flush=True)
        try:
            super().__exit__(exc_type, exc_val, exc_tb)
            self.inner_scheduler.decrease_num_active_tasks()
            #print("Waiting for scheduler to stop", flush=True)

            with self._monitor:
                #print("Busy Waiting for scheduler to stop", flush=True)
                while self.inner_scheduler.get_status():
                    self._monitor.wait()

                #print("Scheduler: Stopping from __exit__", flush=True)
                for t in self.worker_threads:
                    t.join()
        except Exception as e:
            self.exception_stack.append(e)

            if len(self.exception_stack) > 0:
                raise self.exception_stack[0]
        finally:
            pass
            #print("Runtime Stopped", flush=True)

    def run(self):
        #print("Scheduler: Running", flush=True)
        self.inner_scheduler.run()
        #print("Scheduler: Stopped Loop", flush=True)

    def stop(self):
        #print("Scheduler: Stopping (Called from Python)", flush=True)
        self.inner_scheduler.stop()

    def stop_callback(self):
        super().stop()

        for w in self.worker_threads:
            w.stop()

        #print("Scheduler: Stopped", flush=True)

    def spawn_task(self, function, args, dependencies, taskid, req, name):
        nvtx.push_range(message="scheduler::spawn_task", domain="Python Runtime", color="blue")
        self.inner_scheduler.increase_num_active_tasks()
        
        task = ComputeTask(function, args, dependencies, taskid, req, name, scheduler=self)

        #TODO(will): Combine these into an InnerScheduler function that doens't need the GIL
        should_enqueue = task.add_dependencies(task.dependencies)

        if should_enqueue:
            self.inner_scheduler.enqueue_task(task.inner_task)
        nvtx.pop_range(domain="Python Runtime")

        #print("Created Task", task.taskid.full_name, should_enqueue, flush=True)

        return task
    
    def assign_task(self, task, worker):
        #print("Updating task state.", task, worker, flush=True)
        task.state = tasks.TaskRunning(task.func, task.args, task.dependencies)
        #print("Assigning Task", task, worker, flush=True)
        worker.assign_task(task)
        #print("Finished Assigning Task", task, worker, flush=True)

    
def _task_callback(task, body):
    """
    A function which forwards to a python function in the appropriate device context.
    """
    try:
        body = body

        if inspect.iscoroutinefunction(body):
            body = body()

        if inspect.iscoroutine(body):
            try:
                in_value_task = getattr(task, "value_task", None)
                in_value = in_value_task and in_value_task.result

                new_task_info = body.send(in_value)
                task.value_task = None
                if not isinstance(new_task_info, tasks.TaskAwaitTasks):
                    raise TypeError(
                        "Parla coroutine tasks must yield a TaskAwaitTasks")
                dependencies = new_task_info.dependencies
                value_task = new_task_info.value_task
                if value_task:
                    assert isinstance(value_task, Task)
                    task.value_task = value_task
                return tasks.TaskRunning(_task_callback, (body,), dependencies)
            except StopIteration as e:
                result = None
                if e.args:
                    (result,) = e.args
                return tasks.TaskCompleted(result)
        else:
            result = body()
            return tasks.TaskCompleted(result)
    finally:
        pass

    assert False