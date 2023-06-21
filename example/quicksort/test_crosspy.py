

# from parla import Parla, spawn, TaskSpace
import argparse
import cupy as cp
import numpy as np
import crosspy as xp
import math

a = cp.random.rand(10)
b = xp.array(a)
print(b.values()[0][0])
print(list(b.device_view())[0][0])
