from collections import namedtuple, deque

import random


Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward",
                         # We may store information for a GCN layer too.
                         # If any GCN layer is not used, these should be `None`.
                         "gcn_state", "gcn_edgeindex",
                         "next_gcn_state", "next_gcn_edgeindex"))


class ReplayMemory(object):
    """ Experience replay memory.
    This class manages state transition information.
    In DQN, a batch of the transitions are sampled, and are used
    to optimize a model.
    """

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """ Push a new transition.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """ Sample `batch_size` transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """ Return the length of the replay memory.
        """
        return len(self.memory)

    def print(self):
        idx = 0
        for t in self.memory:
            print(idx, " State:", t.state, ", Action:", t.action,
                  ", Next State:", t.next_state, ", Reward:", t.reward)
            idx += 1
