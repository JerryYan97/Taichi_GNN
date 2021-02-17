import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import BatchNorm
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import avg_pool
from torch_geometric.data import Data

# Baseline comparison.
class GCN3D_Feb13_FiveLinears(nn.Module):
    def __init__(self, graph_node_num, cluster_num,
                 nfeat, gcn_hid1, gcn_out1,
                 gcn_hid2, gcn_out2,
                 fc_hid, fc_out, dropout, device):
        super(GCN3D_Feb13_FiveLinears, self).__init__()
        self.GCN1 = GCNConv(nfeat, 256)
        self.fc1 = nn.Linear(256, 256)

        self.istn = InstanceNorm(gcn_out1)

        self.GCN2 = GCNConv(256, 128)
        self.fc2 = nn.Linear(128, 128)

        self.GCN3 = GCNConv(128, 64)
        self.fc3 = nn.Linear(64, 64)

        self.GCN4 = GCNConv(64, 32)
        self.fc4 = nn.Linear(32, 32)

        self.GCN5 = GCNConv(32, 3)
        self.fc5 = nn.Linear(3, 3)

        self.ELU = torch.nn.ELU()
        self.dropout = dropout
        self._device = device
        self._network_out = fc_out
        self._graph_node_num = graph_node_num
        self._cluster_num = cluster_num

    def forward(self, x, adj, num_graphs, in_batch, cluster):
        x = self.ELU(self.GCN1(x, adj))
        x = self.ELU(self.fc1(x))
        y = self.istn(x)
        y = self.ELU(self.GCN2(y, adj))
        y = self.ELU(self.fc2(y))
        z = self.ELU(self.GCN3(y, adj))
        z = self.ELU(self.fc3(z))
        z = self.ELU(self.GCN4(z, adj))
        z = self.ELU(self.fc4(z))
        z = self.ELU(self.GCN5(z, adj))
        z = self.fc5(z)
        return z
