import torch

from dataclasses import dataclass, field

"""
Input data class for network classes such as FCN, GCN, etc.

This class provides a wrapper for network classes.
This is useful since the input format may be different
depending on a network type. For example, a fully-connected
network (FCN) only requires a state, but a graph-convolution
network (GCN) requires not only a state, but also features and a graph topology.
"""
@dataclass
class NetworkInput:
    # State other than vertex feature
    x: torch.Tensor
    # Specify its current input is a batch or not.
    # This is necessary since GCNConv does not support
    # batching like other networks such as FCN; this makes
    # sense since GCN supports minibatching. This just does
    # not match to the DQN model.
    is_batch: bool = False
    # Vertex feature
    gcn_x: torch.Tensor = None
    # Pairs of the edge index in a COO format
    gcn_edge_index: torch.Tensor = None
