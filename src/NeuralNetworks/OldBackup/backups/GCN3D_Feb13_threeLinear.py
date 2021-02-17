import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import BatchNorm
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import avg_pool
from torch_geometric.data import Data

# Baseline comparison.
class GCN3D_Feb13_threeLinear(nn.Module):
    def __init__(self, graph_node_num, cluster_num,
                 nfeat, gcn_hid1, gcn_out1,
                 gcn_hid2, gcn_out2,
                 fc_hid, fc_out, dropout, device):
        super(GCN3D_Feb13_threeLinear, self).__init__()
        self.fc_G1 = nn.Linear(nfeat, 256)
        self.istn = InstanceNorm(256)
        self.fc_G2 = nn.Linear(256, 64)
        self.fc_G3 = nn.Linear(64, 3)

        self.ELU = torch.nn.ELU()
        self.dropout = dropout
        self._device = device
        self._network_out = fc_out
        self._graph_node_num = graph_node_num
        self._cluster_num = cluster_num

    def forward(self, x, adj, num_graphs, in_batch, cluster):
        x = self.ELU(self.fc_G1(x))
        x = self.istn(x)
        x = self.ELU(self.fc_G2(x))
        x = self.ELU(self.fc_G3(x))
        return x
