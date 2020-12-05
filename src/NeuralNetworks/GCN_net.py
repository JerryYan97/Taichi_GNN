import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        # middle layers
        self.conv3 = GCNConv(nhid, nhid)
        self.conv5 = GCNConv(nhid, nclass)
        # self.dropout = dropout

    # Maybe redundent
    # def forward(self, x, adj):
    #     x = self.conv1(x, adj)
    #     x = torch.tanh(x)
    #     x = self.conv3(x, adj)
    #     x = torch.tanh(x)  # -1 -1
    #     x = self.conv3(x, adj)
    #     x = torch.tanh(x)  # -1 -1
    #     # x = self.conv4(x, adj)
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = self.conv5(x, adj)
    #     return x

    # It may has something wrong with the hidden layer
    # 11 10
    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = torch.tanh(x)
        x = self.conv3(x, adj)
        x = torch.tanh(x)  # -1 -1
        # x = self.conv3(x, adj)
        # x = torch.tanh(x)  # -1 -1
        # x = self.conv4(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv5(x, adj)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    # def forward(self, x, adj):
    #     h = self.conv1(x, adj)
    #     h = torch.tanh(h)
    #     h = self.conv3(h, adj)
    #     h = torch.tanh(h)  # -1 -1
    #     h = self.conv3(h, adj)
    #     h = torch.tanh(h)  # -1 -1
    #     # x = self.conv4(x, adj)
    #     h = F.dropout(h, self.dropout, training=self.training)
    #     h = self.conv5(h, adj)
    #     return h