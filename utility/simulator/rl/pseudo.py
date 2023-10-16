import torch
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
