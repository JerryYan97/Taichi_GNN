import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import BatchNorm
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import avg_pool
from torch_geometric.data import Data

# Baseline comparison.
class GCN3D_Feb13(nn.Module):
    def __init__(self, graph_node_num, cluster_num,
                 nfeat, gcn_hid1, gcn_out1,
                 gcn_hid2, gcn_out2,
                 fc_hid, fc_out, dropout, device):
        super(GCN3D_Feb13, self).__init__()
        self.GCN1 = GCNConv(nfeat, 256)
        self.fc1 = nn.Linear(256, 256)

        self.istn = InstanceNorm(gcn_out1)

        self.GCN2 = GCNConv(256, 128)
        self.fc2 = nn.Linear(128, 128)

        self.GCN3 = GCNConv(128, 3)
        self.fc3 = nn.Linear(3, 3)

        self.ELU = torch.nn.ELU()
        self.dropout = dropout
        self._device = device
        self._network_out = fc_out
        self._graph_node_num = graph_node_num
        self._cluster_num = cluster_num



        # self.GCN1 = GCNConv(nfeat, gcn_hid1)
        # self.GCN2 = GCNConv(gcn_hid1, gcn_hid1)
        # self.GCN3 = GCNConv(gcn_hid1, gcn_out1)
        #
        # # self.bn = BatchNorm(gcn_out1)
        # self.istn = InstanceNorm(gcn_out1)
        # self.fc_Temp1 = nn.Linear(gcn_out1, 128)
        # self.fc_Temp2 = nn.Linear(128, 128)
        # self.GCN4 = GCNConv(128, gcn_hid2)
        # self.GCN5 = GCNConv(gcn_hid2, gcn_out2)
        #
        # self.fc1 = nn.Linear(gcn_out2, graph_node_num)
        # self.fc2 = nn.Linear(cluster_num, fc_hid)
        # self.fc3 = nn.Linear(fc_hid, fc_out)
        #
        # self.dropout = dropout
        # self._network_out = fc_out
        # self._graph_node_num = graph_node_num
        # self._cluster_num = cluster_num
        # self.ELU = torch.nn.ELU()


# Backup note:
# PyG's mini-batch simply stacks all the X sequentially.
# Its input x should be [the num of graphs in this batch * the num of node of a graph, feature vector length]
    def forward(self, x, adj, num_graphs, in_batch, cluster):
        x = self.ELU(self.GCN1(x, adj))
        x = self.ELU(self.fc1(x))
        y = self.istn(x)
        y = self.ELU(self.GCN2(y, adj))
        y = self.ELU(self.fc2(y))
        z = self.ELU(self.GCN3(y, adj))
        z = self.fc3(z)
        return z
