import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import BatchNorm
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import avg_pool
from torch_geometric.data import Data

# This is based on the Dec9 network. It just changes its BatchNorm to InstanceNorm.
class GCN3D_Feb12(nn.Module):
    def __init__(self, graph_node_num, cluster_num,
                 nfeat, gcn_hid1, gcn_out1,
                 gcn_hid2, gcn_out2,
                 fc_hid, fc_out, dropout, device):
        super(GCN3D_Feb12, self).__init__()
        self.GCN_G1 = GCNConv(nfeat, 256)
        self.fc_G1 = nn.Linear(256, 256)

        self.istn = InstanceNorm(gcn_out1)

        self.GCN_L1 = GCNConv(256, 128)
        self.fc_L1 = nn.Linear(128, 128)

        self.GCN_O1 = GCNConv(384, 128)
        self.fc_O1 = nn.Linear(128, 3)

        self.ELU = torch.nn.ELU()
        self.dropout = dropout
        self._device = device
        self._network_out = fc_out
        self._graph_node_num = graph_node_num
        self._cluster_num = cluster_num

# Backup note:
# PyG's mini-batch simply stacks all the X sequentially.
# Its input x should be [the num of graphs in this batch * the num of node of a graph, feature vector length]
    def forward(self, x, adj, num_graphs, in_batch, cluster):
        x = self.ELU(self.GCN_G1(x, adj))
        x = self.ELU(self.fc_G1(x))
        y = self.istn(x)

        transformed_batch = in_batch * self._cluster_num
        input_graph = Data(x=y, edge_index=adj, batch=in_batch)
        batch_cluster = cluster + transformed_batch
        compressed_graph = avg_pool(batch_cluster, input_graph)
        local_x = self.ELU(self.GCN_L1(compressed_graph.x, compressed_graph.edge_index))
        local_x = self.ELU(self.fc_L1(local_x))

        idx_mat = torch.transpose(batch_cluster.repeat(128, 1), 0, 1)
        compressed_info = torch.gather(local_x, 0, idx_mat)
        # for i in range(batch_cluster.shape[0]):
        #     compressed_info[i] = local_x[batch_cluster[i]]

        stacked_info = torch.cat((y, compressed_info.to(self._device)), dim=1)
        z = self.ELU(self.GCN_O1(stacked_info, adj))
        z = self.fc_O1(z)
        return z
