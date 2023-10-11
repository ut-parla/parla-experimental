from utility.simulator.rl.models.dqn import *

import torch

agent = DQNAgent(10, 1)
x = torch.zeros(10)
x = agent.select_device(x)
print(x)

