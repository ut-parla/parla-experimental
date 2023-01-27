
from parla import Task, PyInnerTask
import pytest


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


@pytest.mark.parametrize("n_deps", [1, 10, 100])
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


@pytest.mark.parametrize("n_deps", [1, 10, 100])
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


@pytest.mark.parametrize("n_deps", [1, 10, 100])
def test_dependency_notify_serial(n_deps):

    A = Task(name="A")

    dependency_list = []
    for i in range(n_deps):
        task = Task(name=str(i))
        dependency_list.append(task)

    A.add_dependencies(dependency_list)

    for i, task in enumerate(dependency_list):
        task.set_complete()

    assert A.get_num_blocking_dependencies() == 0
