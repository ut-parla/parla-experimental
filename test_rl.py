from utility.simulator.rl.models.dqn import *

import torch

agent = DQNAgent(10, 4)
for i in range(1000):
    x = torch.zeros(10)
    #x = agent.select_device(x)
    x[0] = i
    print(i, ", ",x)

    y = x.clone()
    y[0] += 1

    a = torch.zeros(1, dtype=torch.int64)
    a[0] = 1

    r = torch.zeros(1)
    r[0] = i + 10
    agent.append_transition(x, a, y, r)

#agent.replay_memory.print()
#t = agent.replay_memory.sample(10)
#print(t)

agent.optimize_model()
