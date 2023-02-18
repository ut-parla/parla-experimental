import pytest
from parla.cython.tasks import TaskSpace


def test_element_access():

    n = 100

    T = TaskSpace("T")
    for i in range(n):
        task_list = T[i]
        assert task_list.name == f"T_{i}"

    for i in range(n):
        task_list = T[i]
        assert task_list.name == f"T_{i}"

    assert len(T) == n


def test_multiindex_access():
    n = 100

    T = TaskSpace("T")
    for i in range(n):
        for j in range(n):
            task_list = T[i, j]
            assert task_list.name == f"T_{i}_{j}"

    for i in range(n):
        for j in range(n):
            task_list = T[i, j]
            assert task_list.name == f"T_{i}_{j}"


def test_slice_access():
    n = 10

    T = TaskSpace("T")
    task_list = T[:n]

    for i, task in enumerate(task_list):
        assert task.name == f"T_{i}"


def test_slice_repeat():
    n = 10

    T = TaskSpace("T")
    task_list = T[:n]
    task_list = T[:n]

    for i, task in enumerate(task_list):
        assert task.name == f"T_{i}"


def test_view():
    n = 10
    l = n//2

    T = TaskSpace("T")
    task_list = T[:l]
    task_list = T.view[:n]

    for i, task in enumerate(task_list):
        assert task.name == f"T_{i}"

    assert len(T) == l


def test_bounded():

    l = 10
    m = 5
    n = 4

    T = TaskSpace("T", shape=(m, n))

    task_list = T[:, :]

    print(task_list)

    assert len(task_list) == m*n
    assert len(T) == m*n


def test_default_start():
    n = 10
    l = n//2

    T = TaskSpace("T")

    for i in range(0, n):
        task = T[i-l]
        print(task)

    assert len(T) == l


def test_start():
    n = 10
    l = n//2

    T = TaskSpace("T", start=(-2,))

    for i in range(0, n):
        task = T[i-l]
        print(task)

    assert len(T) == l+2
