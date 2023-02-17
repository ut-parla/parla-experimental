import time
import google_benchmark as benchmark
from google_benchmark import Counter


from parla import parse_index, cy_parse_index
from parla.cython import tasks as cy_tasks
from parla.common import containers


class GetItem:
    def __getitem__(self, index):
        # print(index)
        return index


@benchmark.register(name="PythonParseIndex")
def test(state):
    obj = GetItem()
    index = obj[1, 2:100]
    while state:

        if not isinstance(index, tuple):
            index = (index,)

        ret = []
        def step(x, i): return x + (i,)
        def stop(x): ret.append(x)

        parse_index((), index, step, stop)


@benchmark.register(name="CythonParseIndex")
def test(state):
    obj = GetItem()
    index = obj[1, 2:100]
    while state:

        if not isinstance(index, tuple):
            index = (index,)

        ret = []
        cy_parse_index(ret, (), index)


@benchmark.register(name="PythonTaskID")
def test(state):
    obj = GetItem()
    index = obj[1, 2:100]

    while state:
        data = {}
        ret = containers.make_ids(index, data)


@benchmark.register(name="CythonTaskID")
def test(state):
    obj = GetItem()
    index = obj[1, 2:100]

    while state:
        data = {}
        ret = cy_tasks.make_ids(index, data)


if __name__ == "__main__":
    benchmark.main()
