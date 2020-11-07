import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid*2, nclass)
        # middle layers
        self.conv3 = GCNConv(nhid, nhid)
        self.conv4 = GCNConv(nhid, nhid*2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = torch.tanh(x)
        x = self.conv3(x, adj)
        x = torch.tanh(x)  # -1 -1
        x = self.conv3(x, adj)
        x = torch.tanh(x)  # -1 -1
        x = self.conv4(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x