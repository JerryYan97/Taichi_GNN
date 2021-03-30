import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import BatchNorm
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import avg_pool
from torch_geometric.data import Data


# This is based on Feb14_PoolingDeep NN
# Note: For different mesh, it may needs different width.
# The version here is used for case 1009.
class GCN3D_Mar28_PoolingDeepGlobal(nn.Module):
    def __init__(self, graph_node_num, cluster_num, batch_num,  # The num of batch should be 1.
                 nfeat, fc_out, dropout, device):
        super(GCN3D_Mar28_PoolingDeepGlobal, self).__init__()
        self.global_feat_num = 5 * cluster_num

        self.GCN_G1 = GCNConv(nfeat, 64)
        self.fc_G1 = nn.Linear(64, 128)

        self.GCN_G2 = GCNConv(128, 256)
        self.fc_G2 = nn.Linear(256, 256)

        self.istn = InstanceNorm(256)

        self.GCN_L1 = GCNConv(256, 128)
        self.fc_L1 = nn.Linear(128, 96)

        self.GCN_L2 = GCNConv(96, 64)
        self.fc_L2 = nn.Linear(64, 32)

        self.GCN_M1 = GCNConv(32, 16)
        self.fc_M1 = nn.Linear(16, 8)

        self.GCN_O1 = GCNConv(8, 5)
        self.fc_O1 = nn.Linear(self.global_feat_num, 3 * graph_node_num * batch_num, bias=False)

        self.ELU = torch.nn.ELU()
        self.dropout = dropout
        self._device = device
        self._network_out = fc_out
        self._graph_node_num = graph_node_num
        self._cluster_num = cluster_num

# Backup note:
# PyG's mini-batch simply stacks all the X sequentially.
# Its input x should be [the num of graphs in this batch * the num of node of a graph, feature vector length]
    def forward(self, x, adj, in_batch, cluster):
        x = self.ELU(self.GCN_G1(x, adj))
        x = self.ELU(self.fc_G1(x))
        x = self.ELU(self.GCN_G2(x, adj))
        x = self.ELU(self.fc_G2(x))

        y = self.istn(x)

        transformed_batch = in_batch * self._cluster_num
        batch_cluster = cluster + transformed_batch
        input_graph = Data(x=y, edge_index=adj, batch=in_batch)
        compressed_graph = avg_pool(batch_cluster, input_graph)
        z = self.ELU(self.GCN_L1(compressed_graph.x, compressed_graph.edge_index))
        z = self.ELU(self.fc_L1(z))
        z = self.ELU(self.GCN_L2(z, compressed_graph.edge_index))
        z = self.ELU(self.fc_L2(z))

        z = self.ELU(self.GCN_M1(z, compressed_graph.edge_index))
        z = self.ELU(self.fc_M1(z))
        g_feat = torch.flatten(self.GCN_O1(z, compressed_graph.edge_index))
        # g_feat = self.ELU(self.GCN_O1(z, compressed_graph.edge_index))
        o_feat = self.fc_O1(g_feat)

        return g_feat, o_feat
