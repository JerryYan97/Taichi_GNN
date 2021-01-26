import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import avg_pool
from torch_geometric.data import Data
import time


class GCNOLD(nn.Module):
    def __init__(self, nfeat, graph_node_num, cluster_num,
                 gcn_hid1, gcn_hid2, gcn_hid3,
                 gcn_out1, gcn_out2, gcn_out3,
                 fc_hid, fc_out,
                 num_gcns, num_linear):
        super(GCNOLD, self).__init__()

        self.FC1 = nn.Linear(nfeat, 32)
        self.FC2 = nn.Linear(32, 64)
        self.FC3 = nn.Linear(64, 128)

        self.GCN1 = GCNConv(128, gcn_hid1)
        self.GCN2 = GCNConv(gcn_hid1, gcn_hid1)
        self.GCN3 = GCNConv(gcn_hid1, gcn_hid1)
        self.GCN4 = GCNConv(gcn_hid1, gcn_hid1)
        self.GCN5 = GCNConv(gcn_hid1, gcn_out1)

        self.is_norm = InstanceNorm(gcn_out1)

        self.fc_Temp1 = nn.Linear(gcn_out1, gcn_out1*2)
        self.fc_Temp2 = nn.Linear(gcn_out1*2, gcn_out1*2)
        self.fc_Temp3 = nn.Linear(gcn_out1*2, gcn_out1*2)

        self.GCN6 = GCNConv(gcn_out1*2, gcn_hid2)
        self.GCN7 = GCNConv(gcn_hid2, gcn_hid2)
        self.GCN8 = GCNConv(gcn_hid2, gcn_hid2)
        self.GCN9 = GCNConv(gcn_hid2, gcn_hid2)
        self.GCN10 = GCNConv(gcn_hid2, gcn_out2)

        self.GCN11 = GCNConv(gcn_out2, gcn_hid3)
        self.GCN12 = GCNConv(gcn_hid3, gcn_hid3)
        self.GCN13 = GCNConv(gcn_hid3, gcn_hid3)
        self.GCN14 = GCNConv(gcn_hid3, gcn_hid3)
        self.GCN15 = GCNConv(gcn_hid3, gcn_out3)

        self.fc1 = nn.Linear(gcn_out3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, fc_out)

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self._network_out = fc_out
        self._graph_node_num = graph_node_num
        self._cluster_num = cluster_num
        self.ELU = torch.nn.ELU()
        self.RELU = torch.nn.ReLU()

# Backup note:
# PyG's mini-batch simply stacks all the X sequentially.
# Its input x should be [the num of graphs in this batch * the num of node of a graph, feature vector length]
    def forward(self, x, adj, num_graphs, in_batch, cluster, Up, cluster_parent, cluster_belong):
        # print("x shape: ", x.size())
        # print("adj shape: ", adj.size())
        # print("num graph shape: ", num_graphs)
        # in_batch = torch.squeeze(in_batch, dim=0)
        # cluster = torch.squeeze(cluster, dim=0)
        # print("cluster shape: ", cluster.size())
        # print("in batch shape: ", in_batch.size())
        # time.sleep(3)
        x = self.FC1(x)
        x = self.RELU(x)
        x = self.FC2(x)
        x = self.RELU(x)
        x = self.FC3(x)

        x = self.GCN1(x, adj)
        x = self.ELU(x)
        x = self.GCN2(x, adj)
        x = self.ELU(x)
        x = self.GCN3(x, adj)
        x = self.ELU(x)
        x = self.GCN4(x, adj)
        x = self.ELU(x)
        x = self.GCN5(x, adj)

        y = self.is_norm(x)

        transformed_batch = in_batch * (self._graph_node_num + 1)
        # print("transformed_batch shape: ", transformed_batch.size())
        input_graph = Data(x=y, edge_index=adj, batch=in_batch)
        compressed_graph = avg_pool(cluster + transformed_batch, input_graph)
        # print("x shape 2 : ", compressed_graph.x.size())
        z = self.fc_Temp1(compressed_graph.x)
        z = self.ELU(z)
        z = self.fc_Temp2(z)
        z = self.ELU(z)
        z = self.fc_Temp3(z)

        z = self.GCN6(z, compressed_graph.edge_index)
        z = self.ELU(z)
        z = self.GCN7(z, compressed_graph.edge_index)
        z = self.ELU(z)
        z = self.GCN8(z, compressed_graph.edge_index)
        z = self.ELU(z)
        z = self.GCN9(z, compressed_graph.edge_index)
        z = self.ELU(z)
        z = self.GCN10(z, compressed_graph.edge_index)

        # print("x shape 2 : ", z.size(), " ", z.dtype)   # up sampling
        # print("UP shape : ", Up.size())
        # set the center of the clusters
        for p in range(self._cluster_num):
            batch_list = [(self._graph_node_num*n+cluster_parent[p]) for n in range(0, num_graphs)]
            c_list = [(self._cluster_num*n+p) for n in range(0, num_graphs)]
            Up[batch_list, :] = z[c_list, :]
        # set the cluster children belongs
        for b in range(self._cluster_num):
            # print("z parent size: ", z[b, :].size())
            v = z[b, :]
            Up[cluster_belong[b], :] = v[None]

        z = self.GCN11(Up, adj)
        z = self.ELU(z)
        z = self.GCN12(z, adj)
        z = self.ELU(z)
        z = self.GCN13(z, adj)
        z = self.ELU(z)
        z = self.GCN14(z, adj)
        z = self.ELU(z)
        z = self.GCN15(z, adj)

        # print("x shape 3 : ", z.size())
        k = self.fc1(z)
        k = self.ELU(k)
        k = self.fc2(k)
        k = self.ELU(k)
        k = self.fc3(k)
        k = self.ELU(k)
        k = self.fc4(k)

        return k
