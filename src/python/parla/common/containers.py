from typing import Awaitable, Collection, Iterable
from parla.cython import tasks

TaskAwaitTasks = tasks.TaskAwaitTasks
TaskID = tasks.TaskID

import nvtx

class TaskSet(Awaitable, Collection):

    def __init__(self, tasks):
        self._tasks = tasks

    def _tasks(self):
        pass

    @property
    def _flat_tasks(self):
        dependencies = []
        for ds in self._tasks:
            if not isinstance(ds, Iterable):
                ds = (ds,)
            for d in ds:
                if hasattr(d, "task"):
                    if d.task is not None:
                        d = d.task
                dependencies.append(d)

        return dependencies

    def __await__(self):
        return (yield TaskAwaitTasks(self._flat_tasks, None))

    def __len__(self):
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)

    def __contains__(self, x):
        return x in self._tasks

    def __repr__(self):
        return "tasks({})".format(", ".join(repr(x) for x in self._tasks))


class Tasks(TaskSet):

    @property
    def _tasks(self):
        return self.args

    def __init__(self, *args):
        self.args = args


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


class TaskSpace(TaskSet):

    @property
    def _tasks(self):
        return self._data.values()

    def __init__(self, name="", members=None):
        self._data = members or {}
        self._name = name

    def __getitem__(self, index):

        if not isinstance(index, tuple):
            index = (index,)
        ret = []

        parse_index((), index, lambda x, i: x + (i,),
                    lambda x: ret.append(self._data.setdefault(x, TaskID(self._name, x))))
        # print("index ret", ret, flush=True)
        if len(ret) == 1:
            return ret[0]
        return ret

    def __repr__(self):
        return "TaskSpace({self._name}, {_data})".format(**self.__dict__)
