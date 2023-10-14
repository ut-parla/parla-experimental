import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

"""
3 Graph Convolutional Networks (GCNs) + 1 Fully-Conneceted Network (FCN).
"""
class GCN(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.gcn_indim = in_dim
        self.outdim = out_dim
        self.gcn = GCNConv(self.gcn_indim, self.gcn_indim, device=self.device, flow="target_to_source").to(device=self.device)
        self.out = Linear(self.gcn_indim, self.outdim, device=self.device)

    def forward(self, x, ei):
        print("input x:", x, " ei:", ei)
        x = self.gcn(x, ei)
        print("bias:", self.gcn.bias)
        for param in self.gcn.lin.parameters():
            print("param:", param)
        print("out x list:", x)
        x = torch.mean(x, dim=0)
        print("out x:", x)
        x = F.log_softmax(self.out(x), 0)
        # If `out_dim` is 1, only one action is available.
        # Do not squeeze the output in this case.
        return x if self.outdim == 1 else x.squeeze(0)
