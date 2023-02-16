import time
import threading
import pytest

from parla.cython import tasks as cy_tasks


class GetItem:
    def __getitem__(self, index):
        return index


def test_parse_index():
    obj = GetItem()
    index = obj[1, 2:100]
    data = {}

    print(index)
    ret = cy_tasks.make_ids(index, data)
    print(ret)


test_parse_index()
