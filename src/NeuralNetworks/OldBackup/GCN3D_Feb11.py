import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import BatchNorm
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import avg_pool
from torch_geometric.data import Data


# This is based on the Dec9 network. It just changes its BatchNorm to InstanceNorm.
class GCN3D_Feb11(nn.Module):
    def __init__(self, graph_node_num, cluster_num,
                 nfeat, gcn_hid1, gcn_out1,
                 gcn_hid2, gcn_out2,
                 fc_hid, fc_out, dropout):
        super(GCN3D_Feb11, self).__init__()
        self.GCN1 = GCNConv(nfeat, gcn_hid1)
        self.GCN2 = GCNConv(gcn_hid1, gcn_hid1)
        self.GCN3 = GCNConv(gcn_hid1, gcn_out1)
        self.in_lins = torch.nn.ModuleList()
        self.in_lins.append(nn.Linear(gcn_out1, gcn_out1))
        self.in_lins.append(nn.Linear(gcn_out1, gcn_out1))

        # self.bn = BatchNorm(gcn_out1)
        self.istn = InstanceNorm(gcn_out1)
        self.fc_Temp1 = nn.Linear(gcn_out1, 128)
        self.fc_Temp2 = nn.Linear(128, 128)
        self.GCN4 = GCNConv(128, gcn_hid2)
        self.GCN5 = GCNConv(gcn_hid2, gcn_out2)

        self.fc1 = nn.Linear(gcn_out2, graph_node_num)
        self.before_view_lins = torch.nn.ModuleList()
        self.before_view_lins.append(nn.Linear(graph_node_num, graph_node_num))
        self.before_view_lins.append(nn.Linear(graph_node_num, graph_node_num))

        self.fc2 = nn.Linear(cluster_num, fc_hid)
        self.fc3 = nn.Linear(fc_hid, fc_out)

        self.after_view_lins = torch.nn.ModuleList()
        self.after_view_lins.append(nn.Linear(fc_out, fc_out))
        self.after_view_lins.append(nn.Linear(fc_out, fc_out))

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
        x = self.ELU(self.in_lins[0](x))
        x = self.ELU(self.in_lins[1](x))

        # y = self.bn(x)
        y = self.istn(x)

        transformed_batch = in_batch * (self._graph_node_num + 1)

        input_graph = Data(x=y, edge_index=adj, batch=in_batch)
        compressed_graph = avg_pool(cluster + transformed_batch, input_graph)
        z = self.fc_Temp1(compressed_graph.x)
        z = self.ELU(z)
        z = self.fc_Temp2(z)
        z = self.ELU(z)
        z = self.GCN4(z, compressed_graph.edge_index)
        z = self.ELU(z)
        z = self.GCN5(z, compressed_graph.edge_index)
        z = self.ELU(z)

        z = self.ELU(self.fc1(z))
        z = self.ELU(self.before_view_lins[0](z))
        z = self.ELU(self.before_view_lins[1](z))

        k = z.view(-1, self._cluster_num)

        k = self.fc2(k)
        k = self.ELU(k)
        k = self.fc3(k)
        k = self.ELU(self.after_view_lins[0](k))
        k = self.ELU(self.after_view_lins[1](k))

        return k
