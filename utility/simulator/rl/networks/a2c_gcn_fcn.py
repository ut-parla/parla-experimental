import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


"""
3 Graph Convolutional Networks (GCNs) + 1 Fully-Conneceted Network (FCN).
"""
class A2CNetwork(torch.nn.Module):

    def __init__(self, gcn_indim: int, in_dim: int, out_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.gcn_indim = gcn_indim
        self.fcn1_indim = in_dim
        self.fcn1_outdim = in_dim * 4
        self.fcn2_outdim = in_dim * 8
        self.actor_outdim = out_dim
        self.critic_outdim = 1
        self.gcn = GCNConv(self.gcn_indim, self.gcn_indim).to(device=self.device)

        # Actor configuration
        self.actor_fcn1 = Linear(self.fcn1_indim, self.fcn1_outdim,
                                 device=self.device)
        self.actor_fcn2 = Linear(self.fcn1_outdim, self.fcn2_outdim,
                                 device=self.device)
        self.actor_out = Linear(self.fcn2_outdim, self.actor_outdim,
                                device=self.device)
        # Critic configuration
        self.critic_fcn1 = Linear(self.fcn1_indim, self.fcn1_outdim,
                                  device=self.device)
        self.critic_fcn2 = Linear(self.fcn1_outdim, self.fcn2_outdim,
                                  device=self.device)
        self.critic_out = Linear(self.fcn2_outdim, self.critic_outdim,
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
                # Only aggregate gcn output tensors.
                gcn_out = torch.mean(gcn_out, dim=0)
                y = torch.cat((gcn_out, x[i]), dim=0)
                lst_y.append(y)
            x = torch.stack(lst_y)
        else:
            gcn_out = self.gcn(gcn_x, gcn_edge_index)
            gcn_out = torch.mean(gcn_out, dim=0)
            # Concatenate a gcn tensor and a normal state.
            # Expected dimension: [gcn tensor values, normal state values]
            x = torch.cat((gcn_out, x), dim=0)
        # Actor forward
        a = F.leaky_relu(self.actor_fcn1(x))
        a = F.leaky_relu(self.actor_fcn2(a))
        a = self.actor_out(a)

        # Critic forward
        c = F.leaky_relu(self.critic_fcn1(x))
        c = F.leaky_relu(self.critic_fcn2(c))
        c = self.critic_out(c)

        return a,c
