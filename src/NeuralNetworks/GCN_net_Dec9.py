import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import avg_pool
from torch_geometric.data import Data


class GCN_net_Dec9(nn.Module):
    def __init__(self, graph_node_num, cluster_num,
                 nfeat, gcn_hid1, gcn_out1,
                 gcn_hid2, gcn_out2,
                 fc_hid, fc_out, dropout):
        super(GCN_net_Dec9, self).__init__()
        self.GCN1 = GCNConv(nfeat, gcn_hid1)
        self.GCN2 = GCNConv(gcn_hid1, gcn_hid1)
        self.GCN3 = GCNConv(gcn_hid1, gcn_out1)

        self.bn = BatchNorm(gcn_out1)

        self.GCN4 = GCNConv(gcn_out1, gcn_hid2)
        self.GCN5 = GCNConv(gcn_hid2, gcn_out2)

        self.fc1 = nn.Linear(gcn_out2, graph_node_num)
        self.fc2 = nn.Linear(cluster_num, fc_hid)
        self.fc3 = nn.Linear(fc_hid, fc_out)

        self.dropout = dropout
        self._network_out = fc_out
        self._graph_node_num = graph_node_num
        self._cluster_num = cluster_num
        self.ELU = torch.nn.ELU()

# Backup note:
# PyG's mini-batch simply stacks all the X sequentially.
# Its input x should be [the num of graphs in this batch * the num of node of a graph, feature vector length]
    def forward(self, x, adj, num_graphs, in_batch, cluster):
        x = self.GCN1(x, adj)
        x = self.ELU(x)
        x = self.GCN2(x, adj)
        x = self.ELU(x)
        x = self.GCN3(x, adj)

        y = self.bn(x)

        input_graph = Data(x=y, edge_index=adj, batch=in_batch)
        avg_pool(cluster, input_graph)

        z = self.GCN4(input_graph.x, input_graph.edge_index)
        z = self.GCN5(z, input_graph.edge_index)

        z = self.fc1(z)
        z = self.ELU(z)

        z = z.view(-1, self._cluster_num)

        z = self.fc2(z)
        z = self.fc3(z)

        return z
