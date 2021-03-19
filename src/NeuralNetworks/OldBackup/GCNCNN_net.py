import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_CNN(nn.Module):
    def __init__(self, nfeat, nhid, nnode, gcnout, cnnout, dropout):
        super(GCN_CNN, self).__init__()
        self.GCN1 = GCNConv(nfeat, nhid)
        self.GCN3 = GCNConv(nhid, nhid)     # middle layers
        self.GCN5 = GCNConv(nhid, gcnout)   # output channel is a int number

        self.CNN1 = nn.Conv2d(1, 6, 5)  # in_channels, out_channels, kernel_size
        self.CNN2 = nn.Conv2d(6, 16, 3)  #
        # self.CNN3 = nn.Conv2d(16, 32, 3)  #
        self.fc1 = nn.Linear(16 * 225 * 14, 60)
        self.fc2 = nn.Linear(60, 32)
        self.fc3 = nn.Linear(32, cnnout)

        self.dropout = dropout
        self._gcnout = gcnout
        self._nnode = nnode

    def forward(self, x, adj, num_graphs):
        x = self.GCN1(x, adj)
        x = torch.tanh(x)
        x = self.GCN3(x, adj)
        x = torch.tanh(x)
        x = self.GCN3(x, adj)
        x = torch.tanh(x)
        x = self.GCN5(x, adj)

        # Batch size, CNN1 in channels, height, width
        z = x.view(num_graphs, 1, self._nnode, self._gcnout)

        y = self.CNN1(z)
        y = self.CNN2(y)
        y = y.view(-1, 16 * 225 * 14)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y
