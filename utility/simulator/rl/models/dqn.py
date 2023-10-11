import torch
import torch.nn

from ..networks.fcn import *

class DQNAgent:

    def __init__(self, in_dim: int, out_dim: int):
        self.network = FCN(in_dim, out_dim)

    def select_device(self, x):
        return self.network(x)
