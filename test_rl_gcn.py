from utility.simulator.rl.models.dqn import *
from utility.simulator.rl.networks.gcn import *
from utility.simulator.rl.networks.gcn_fcn import *

import torch

input_dim = 2
agent = DQNAgent(2, 4, 2)
st = PseudoTask()
st.creates_dummy_task()
tg = TaskGraph()
tg.construct_embedding(st, 2)
gcn = GCN_FCN_Type1(2, 4, 2)
x = torch.zeros(2)
x[0] = 1
x[1] = 2
print("tgx:", tg.x)
print("tg ei:", tg.edge_index)
model_input = NetworkInput(x, False, tg.x, tg.edge_index)
x = gcn(model_input)
print("x:", x)

print("Batching..")
for i in range(1000):
    x = torch.zeros(2)
    x[0] = i
    x[1] = i + 1
    y = x.clone()
    y[0] = i * 10
    y[1] = (i + 1) * 10
    a = torch.zeros(1, dtype=torch.int64)
    a[0] = 1
    r = torch.zeros(1)
    r[0] = i * 1000
    agent.append_transition(x, a, y, r, tg.x, tg.edge_index, tg.x, tg.edge_index)

agent.replay_memory.print()
t = agent.replay_memory.sample(10)
print("T")
print(t)

agent.start_episode()
agent.finalize_episode()
agent.print_model("before_optimization")
agent.optimize_model()
agent.print_model("optimization")
agent.load_models()
agent.print_model("load")

