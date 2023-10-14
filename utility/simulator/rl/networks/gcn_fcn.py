import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


# TODO(hc): This is a pseudo task that is designed for a test.
class PseudoTask:

    def __init__(self):
        self.dependencies = []


    def creates_dummy_task(self):
        self.dependencies.append(PseudoTask())
        self.dependencies.append(PseudoTask())
        self.dependencies.append(PseudoTask())


class TaskGraph(Data):

    def __init__(self):
        Data.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        # TODO(hc): construct an edge index in the COO format.
        #           (2, num_edges) and [0]: src nodes, and [1]: dst nodes.
        # TODO(hc): construct a node array (data.x)
        #           its shape should be (num_nodes, feature_length)

    def construct_embedding(self, task, in_dim):
        #TODO(hc): For now, use a fake task instance. Later, adopt
        #          the real task instance.
        lst_node_features = []
        lst_src_edge_index = []
        lst_dst_edge_index = []
        feature = []
        for i in range(in_dim - 1):
            feature.append(0)
        feature.append(0)
        lst_node_features.append(feature)
        feature_idx = 1
        print("Num dependencies:", task.dependencies)
        for dependency in task.dependencies:
            feature = []
            for i in range(in_dim - 1):    
                feature.append(0)
            feature.append(feature_idx)
            lst_node_features.append(feature)
            lst_src_edge_index.append(feature_idx)
            lst_dst_edge_index.append(0)
            feature_idx += 1
        self.edge_index = torch.Tensor([lst_src_edge_index, lst_dst_edge_index])
        self.edge_index = self.edge_index.to(torch.int64).to(device=self.device)
        self.x = torch.Tensor(lst_node_features).to(device=self.device)


"""
3 Graph Convolutional Networks (GCNs) + 1 Fully-Conneceted Network (FCN).
"""
class GCN_FCN_Type1(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.gcn_indim = in_dim / 2
        self.fcn1_indim = in_dim
        self.fcn1_outdim = in_dim * 4
        self.fcn2_outdim = in_dim * 8
        self.outdim = out_dim
        self.gcn = GCNConv(10, 10, device=self.device).to(device=self.device)
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
        print("gcn_x:", gcn_x, " and ", gcn_edge_index)
        print("norm state:", x)
        if is_batch:
            lst_y = []
            for i in range(len(gcn_x)):
                gcn_out = self.gcn(gcn_x[i], gcn_edge_index[i])
                # Only aggregate GCN output tensors.
                gcn_out = torch.mean(gcn_out, dim=0)
                print("Gcn out:", gcn_out)
                print("x[i]:", x[i])
                y = torch.cat((gcn_out, x[i]), dim=0)
                print("y:", y)
                lst_y.append(y)
            print("lst y:", lst_y)
            x = torch.stack(lst_y)
            print("out x:", x)
        else:
            gcn_out = self.gcn(gcn_x, gcn_edge_index)
            gcn_out = torch.mean(gcn_out, dim=0)
            # Concatenate a GCN tensor and a normal state.
            # Expected dimension: [GCN tensor values, normal state values]
            x = torch.cat((gcn_out, x), dim=0)
            print("out x:", x)
        x = F.leaky_relu(self.fcn1(x))
        x = F.leaky_relu(self.fcn2(x))
        x = F.log_softmax(self.out(x), 0)
        # If `out_dim` is 1, only one action is available.
        # Do not squeeze the output in this case.
        return x if self.outdim == 1 else x.squeeze(0)
