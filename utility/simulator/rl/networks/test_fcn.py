from fcn import *
import torch

fcn = FCN(10, 4)
state = torch.zeros(10)
state[0] = 3
state[3] = 2

print("state=", state)
x = fcn(state)

print("Starts")
print("X=", x)
