import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv
# from torch_geometric.nn import BatchNorm
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import avg_pool
from torch_geometric.data import Data


# This is based on the Dec9 network. It Changes GCNConv to GCN2Conv
# It basically follows:
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn2_cora.py
# and the NN framework in Learning to simulate Complex Physical Simulation.
# class GCN3D_Jan15(nn.Module):
#     def __init__(self, graph_node_num, cluster_num,
#                  nfeat, gcn_hid1, gcn_out1,
#                  gcn_hid2, gcn_out2,
#                  fc_hid, fc_out, dropout):
#         super(GCN3D_Jan15, self).__init__()
#         self.GCN1 = GCNConv(nfeat, gcn_hid1)
#         self.GCN2 = GCNConv(gcn_hid1, gcn_hid1)
#         self.GCN3 = GCNConv(gcn_hid1, gcn_out1)
#
#         # self.bn = BatchNorm(gcn_out1)
#         self.istn = InstanceNorm(gcn_out1)
#         self.fc_Temp1 = nn.Linear(gcn_out1, 32)
#         self.fc_Temp2 = nn.Linear(32, 128)
#         self.GCN4 = GCNConv(128, gcn_hid2)
#         self.GCN5 = GCNConv(gcn_hid2, gcn_out2)
#
#         self.fc1 = nn.Linear(gcn_out2, graph_node_num)
#         self.fc2 = nn.Linear(cluster_num, fc_hid)
#         self.fc3 = nn.Linear(fc_hid, fc_out)
#
#         self.dropout = dropout
#         self._network_out = fc_out
#         self._graph_node_num = graph_node_num
#         self._cluster_num = cluster_num
#         self.ELU = torch.nn.ELU()
#
# # Backup note:
# # PyG's mini-batch simply stacks all the X sequentially.
# # Its input x should be [the num of graphs in this batch * the num of node of a graph, feature vector length]
#     def forward(self, x, adj, num_graphs, in_batch, cluster):
#         x = self.GCN1(x, adj)
#         x = self.ELU(x)
#         x = self.GCN2(x, adj)
#         x = self.ELU(x)
#         x = self.GCN3(x, adj)
#
#         # y = self.bn(x)
#         y = self.istn(x)
#
#         transformed_batch = in_batch * (self._graph_node_num + 1)
#
#         input_graph = Data(x=y, edge_index=adj, batch=in_batch)
#         compressed_graph = avg_pool(cluster + transformed_batch, input_graph)
#         z = self.fc_Temp1(compressed_graph.x)
#         z = self.ELU(z)
#         z = self.fc_Temp2(z)
#         z = self.ELU(z)
#         z = self.GCN4(z, compressed_graph.edge_index)
#         z = self.ELU(z)
#         z = self.GCN5(z, compressed_graph.edge_index)
#         z = self.ELU(z)
#
#         z = self.fc1(z)
#
#         k = z.view(-1, self._cluster_num)
#
#         k = self.fc2(k)
#         k = self.ELU(k)
#         k = self.fc3(k)
#
#         return k

# Note:
# Basic ideas:
# First, we increase the feature num to linear_hid2.
# Then, several GCNII layers are applied to the data.
# Finally, we decrease the feature num to fc_out through two linear layers.
class GCN3D_Jan15(nn.Module):
    def __init__(self, num_layers, alpha, theta, graph_node_num, nfeat, fc_out, dropout,
                 shared_weights=True, linear_hid1=96, linear_hid2=128, linear_hid3=16):
        super(GCN3D_Jan15, self).__init__()

        self.istn = InstanceNorm(nfeat)

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(nfeat, linear_hid1))
        self.lins.append(nn.Linear(linear_hid1, linear_hid2))
        self.lins.append(nn.Linear(linear_hid2, linear_hid3))
        self.lins.append(nn.Linear(linear_hid3, fc_out))

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(linear_hid2, alpha, theta, layer + 1, shared_weights, normalize=False)
            )
        self.dropout = dropout
        self._graph_node_num = graph_node_num
        self.ELU = torch.nn.ELU()

    def forward(self, x, adj, num_graphs, in_batch, cluster):

        x = self.istn(x)

        x = self.lins[0](x)
        x = self.ELU(x)
        x = self.lins[1](x)
        x = x_0 = self.ELU(x)

        for conv in self.convs:
            x = conv(x, x_0, adj)
            x = self.ELU(x)

        x = self.lins[2](x)
        x = self.ELU(x)
        x = self.lins[3](x)
        x = self.ELU(x)

        return x

