from utility.simulator.rl.models.dqn import *

import torch

agent = DQNAgent(20, 4)
st = PseudoTask()
st.creates_dummy_task()
tg = TaskGraph()
tg.construct_embedding(st, 10)
for i in range(1000):
    x = torch.zeros(10)
    x[0] = i
    x = agent.select_device(x, tg.x, tg.edge_index)
    print("x:", x)

    """
    y = x.clone()
    y[0] += 1

    a = torch.zeros(1, dtype=torch.int64)
    a[0] = 1

    r = torch.zeros(1)
    r[0] = i + 10
    agent.append_transition(x, a, y, r, tg.x, tg.edge_index)

agent.replay_memory.print()
t = agent.replay_memory.sample(10)
print(t)

agent.start_episode()
agent.finalize_episode()
#agent.optimize_model()
agent.print_model("optimization")
agent.load_models()
agent.print_model("load")
"""
