from utility.simulator.rl.models.dqn import *

import torch

agent = DQNAgent(10, 4)
for i in range(100):
    x = torch.zeros(10)
    x = agent.select_device(x)
    print(i, ", ",x)

