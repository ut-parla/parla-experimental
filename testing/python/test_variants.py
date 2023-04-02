import pytest
from parla import Parla, spawn
from parla import TaskSpace
from parla import specialize
from parla.utility.execute import timeout
from parla.cython.device_manager import cpu, gpu


@specialize
def f():
    return "DEFAULT"


@f.variant((gpu))
def f_gpu1():
    return "1 GPU"


@f.variant((gpu, gpu))
def f_gpu2():
    return "2 GPU"


@f.variant((gpu, gpu, gpu))
def f_gpu3():
    return "3 GPU"


@f.variant((cpu, gpu))
def f_mix():
    return "1 CPU, 1 GPU"


@f.variant((cpu, gpu, gpu))
def f_mix2():
    return "1 CPU, 2 GPU"


@pytest.mark.parametrize("input,expected", [(cpu, "DEFAULT"), ((gpu, gpu), "2 GPU"), ((cpu, gpu, gpu), "1 CPU, 2 GPU")])
def test_single_task(input, expected):

    @timeout(10)
    def single_task():
        T = TaskSpace("T")
        with Parla(dev_config_file="devices_sample.YAML"):

            @spawn(T[1], placement=[input])
            def task():
                s = f()
                print("VARIANT: ", s, flush=True)
                assert (s == expected)

    single_task()
