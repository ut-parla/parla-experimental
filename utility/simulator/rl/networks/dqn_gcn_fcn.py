import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


"""
3 Graph Convolutional Networks (GCNs) + 1 Fully-Conneceted Network (FCN).
"""
class DQNNetwork(torch.nn.Module):

    def __init__(self, gcn_indim: int, in_dim: int, out_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.gcn_indim = gcn_indim
        self.fcn1_indim = in_dim
        self.fcn1_outdim = in_dim * 4
        self.fcn2_outdim = in_dim * 8
        self.outdim = out_dim
        self.gcn = GCNConv(self.gcn_indim, self.gcn_indim, device=self.device).to(device=self.device)
        self.fcn1 = Linear(self.fcn1_indim, self.fcn1_outdim,
                                   device=self.device)
        self.fcn2 = Linear(self.fcn1_outdim, self.fcn2_outdim,
                                   device=self.device)
        self.out = Linear(self.fcn2_outdim, self.outdim,
                                   device=self.device)

    def forward(self, model_input):
        is_batch = model_input.is_batch
        x = model_input.x.to(self.device)
        gcn_x = model_input.gcn_x.to(self.device)
        gcn_edge_index = model_input.gcn_edge_index.to(self.device)
        if is_batch:
            lst_y = []
            for i in range(len(gcn_x)):
                gcn_out = self.gcn(gcn_x[i], gcn_edge_index[i])
                # Only aggregate GCN output tensors.
                gcn_out = torch.mean(gcn_out, dim=0)
                y = torch.cat((gcn_out, x[i]), dim=0)
                lst_y.append(y)
            x = torch.stack(lst_y)
        else:
            gcn_out = self.gcn(gcn_x, gcn_edge_index)
            gcn_out = torch.mean(gcn_out, dim=0)
            # Concatenate a GCN tensor and a normal state.
            # Expected dimension: [GCN tensor values, normal state values]
            x = torch.cat((gcn_out, x), dim=0)
        x = F.leaky_relu(self.fcn1(x))
        x = F.leaky_relu(self.fcn2(x))
        x = F.log_softmax(self.out(x), 0)
        # If `out_dim` is 1, only one action is available.
        # Do not squeeze the output in this case.
        return x if self.outdim == 1 else x.squeeze(0)
