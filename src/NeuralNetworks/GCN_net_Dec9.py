import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GraphUNet
from torch_geometric.data import Data

class GCN_net_Dec9(nn.Module):
    def __init__(self, graph_node_num,
                 nfeat, gcn_hid1, gcn_out,
                 unet_hid, unet_out,
                 fc_hid, fc_out, dropout):
        super(GCN_net_Dec9, self).__init__()
        self.GCN1 = GCNConv(nfeat, gcn_hid1)
        self.GCN2 = GCNConv(gcn_hid1, gcn_hid1)
        self.GCN3 = GCNConv(gcn_hid1, gcn_out)

        self.bn = BatchNorm(gcn_out)

        # NOTE: According to UNet paper, the depth of 4 can bring better performance than other cases.
        self.GraphUNet = GraphUNet(gcn_out, unet_hid, unet_out, 4)

        self.fc1 = nn.Linear(unet_out, fc_hid)
        self.fc2 = nn.Linear(fc_hid, fc_hid)
        self.fc3 = nn.Linear(fc_hid, fc_out)

        self.dropout = dropout
        self._network_out = fc_out
        self._graph_node_num = graph_node_num
        self.ELU = torch.nn.ELU()

# Backup note:
# PyG's mini-batch simply stacks all the X sequentially.
# Its input x should be [the num of graphs in this batch * the num of node of a graph, feature vector length]
    def forward(self, x, adj, num_graphs, in_batch):
        x = self.GCN1(x, adj)
        x = self.ELU(x)
        x = self.GCN2(x, adj)
        x = self.ELU(x)
        x = self.GCN3(x, adj)

        y = self.bn(x)

        unet_res = self.GraphUNet(y, adj, in_batch)

        z = self.fc1(unet_res)
        z = self.ELU(z)
        z = self.fc2(z)
        z = self.fc3(z)

        return z
