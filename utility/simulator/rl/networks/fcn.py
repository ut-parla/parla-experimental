import torch
import torch.nn
import torch.nn.functional as F


"""
Fully-conneceted network.
"""
class FCN(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.fc1_indim = in_dim
        self.fc1_outdim = in_dim * 4
        self.fc2_outdim = in_dim * 8
        self.outdim = out_dim
        self.fc1 = torch.nn.Linear(self.fc1_indim, self.fc1_outdim,
                                   device=self.device)
        self.fc2 = torch.nn.Linear(self.fc1_outdim, self.fc2_outdim,
                                   device=self.device)
        self.out = torch.nn.Linear(self.fc2_outdim, self.outdim,
                                   device=self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.log_softmax(self.out(x), 0)
        # If `out_dim` is 1, only one action is available.
        # Do not squeeze the output in this case.
        return x if self.outdim == 1 else x.squeeze(0)
