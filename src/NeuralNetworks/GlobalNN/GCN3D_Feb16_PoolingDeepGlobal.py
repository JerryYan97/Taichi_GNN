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
class GCN3D_Feb16_PoolingDeepGlobal(nn.Module):
    def __init__(self, graph_node_num, cluster_num,
                 nfeat, fc_out, dropout, device):
        super(GCN3D_Feb16_PoolingDeepGlobal, self).__init__()
        self.GCN_G1 = GCNConv(nfeat, 64)
        self.fc_G1 = nn.Linear(64, 64)

        self.GCN_G2 = GCNConv(64, 256)
        self.fc_G2 = nn.Linear(256, 256)

        self.istn = InstanceNorm(256)

        self.GCN_L1 = GCNConv(256, 128)
        self.fc_L1 = nn.Linear(128, 128)

        self.GCN_L2 = GCNConv(128, 64)
        self.fc_L2 = nn.Linear(64, 64)

        self.GCN_M1 = GCNConv(320, 128)
        self.fc_M1 = nn.Linear(128, 128)

        self.GCN_M2 = GCNConv(128, 32)
        self.fc_M2 = nn.Linear(32, 32)

        self.GCN_M3 = GCNConv(32, 3)
        self.fc_M3 = nn.Linear(3, 3)

        self.GCN_O1 = GCNConv(3, 3)
        self.fc_O1 = nn.Linear(3, 3)

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
        x = self.ELU(self.GCN_G2(x, adj))
        x = self.ELU(self.fc_G2(x))

        y = self.istn(x)

        transformed_batch = in_batch * self._cluster_num
        batch_cluster = cluster + transformed_batch
        input_graph = Data(x=y, edge_index=adj, batch=in_batch)
        compressed_graph = avg_pool(batch_cluster, input_graph)
        local_x = self.ELU(self.GCN_L1(compressed_graph.x, compressed_graph.edge_index))
        local_x = self.ELU(self.fc_L1(local_x))
        local_x = self.ELU(self.GCN_L2(local_x, compressed_graph.edge_index))
        local_x = self.ELU(self.fc_L2(local_x))

        idx_mat = torch.transpose(batch_cluster.repeat(64, 1), 0, 1)
        compressed_info = torch.gather(local_x, 0, idx_mat)

        stacked_info = torch.cat((y, compressed_info.to(self._device)), dim=1)
        z = self.ELU(self.GCN_M1(stacked_info, adj))
        z = self.ELU(self.fc_M1(z))
        z = self.ELU(self.GCN_M2(z, adj))
        z = self.ELU(self.fc_M2(z))
        z = self.ELU(self.GCN_M3(z, adj))
        z = self.ELU(self.fc_M3(z))

        large_output_graph = Data(x=z, edge_index=adj, batch=in_batch)
        output_graph = avg_pool(batch_cluster, large_output_graph)
        z = self.ELU(self.GCN_O1(output_graph.x, output_graph.edge_index))
        z = self.ELU(self.fc_O1(z))
        idx_mat_output = torch.transpose(batch_cluster.repeat(3, 1), 0, 1)
        output_info = torch.gather(z, 0, idx_mat_output)
        return output_info, z
