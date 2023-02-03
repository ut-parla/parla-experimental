
import time
import threading
import pytest
from parla.cython import tasks
Task = tasks.Task

dep_count = [1, 10, 100]


class Propogate(threading.Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args)
        self.ex = None

    def run(self):
        try:
            self._target(self._args[0])
        except BaseException as e:
            self.ex = e

    def join(self):
        super().join()
        if self.ex is not None:
            raise self.ex


def test_task_creation():
    # Create a task and fetch its own pointer from the C++ InnerTask
    A = Task(name="A")
    B = A.inner_task.get_py_task()

    # Check that they have the same name
    assert A.name == "A"
    assert B.name == "A"

    # Check that they are the same object
    B.name = "B"
    assert A.name == "B"


@pytest.mark.parametrize("n_deps", dep_count)
def test_add_dependency_add_incomplete(n_deps):
    A = Task(name="A")

    dependency_list = []
    for i in range(n_deps):
        task = Task(name=str(i))
        dependency_list.append(task)

    A.add_dependencies(dependency_list)

    # Check that the number of dependencies is correct
    assert A.get_num_dependencies() == n_deps
    assert A.get_num_blocking_dependencies() == n_deps

    # Check that the stored values are correct and retrievable
    output_deps = A.get_dependencies()
    assert len(output_deps) == n_deps
    for i, task in enumerate(output_deps):
        assert isinstance(task, Task)
        assert (task.name == str(i))

    # Check the dependents of each dependency
    for i, task in enumerate(dependency_list):
        assert (task.get_num_dependents() == 1)
        dependents_list = task.get_dependents()

        assert len(task.get_dependents()) == 1

        dependent = dependents_list[0]
        assert isinstance(dependent, Task)
        assert (dependent.name == "A")


@pytest.mark.parametrize("n_deps", dep_count)
def test_add_dependency_add_complete(n_deps):

    A = Task(name="A")

    dependency_list = []
    for i in range(n_deps):
        task = Task(name=str(i))
        task.set_complete()
        dependency_list.append(task)

    A.add_dependencies(dependency_list)

    assert A.get_num_dependencies() == n_deps
    assert A.get_num_blocking_dependencies() == 0


@pytest.mark.parametrize("n_deps", dep_count)
def test_dependency_notify_serial(n_deps):

    A = Task(name="A")

    dependency_list = []
    for i in range(n_deps):
        task = Task(name=str(i))
        dependency_list.append(task)

    A.add_dependencies(dependency_list)

    for i, task in enumerate(dependency_list):
        status = task.notify_dependents_wrapper()
        if i == n_deps - 1:
            assert status == 1
        else:
            assert status == 0

    assert A.get_num_blocking_dependencies() == 0


@pytest.mark.parametrize("n_deps", dep_count)
def test_dependency_notify_parallel(n_deps):
    # TODO: Add randomization so this is a more robust test
    # Create a task with dependencies while those dependencies are completing

    A = Task(name="A")

    delay_notify = 0.001
    delay_create = 0.001

    notify_status_list = []
    create_status_list = []

    dependency_list = []
    for i in range(n_deps):
        task = Task(name=str(i))
        dependency_list.append(task)

    def notify_task(status_list):
        for i, task in enumerate(dependency_list):
            status = task.notify_dependents_wrapper()
            status_list.append(status)

    def create_task(status_list):
        status = A.add_dependencies(dependency_list)
        status_list.append(status)

    thread1 = Propogate(target=notify_task, args=(notify_status_list,))
    thread2 = Propogate(target=create_task, args=(create_status_list,))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # Check that the number of completed dependencies is corrects
    assert A.get_num_blocking_dependencies() == 0

    # Only one of these creation or notify steps should have launched the task
    assert sum(notify_status_list + create_status_list) == 1
