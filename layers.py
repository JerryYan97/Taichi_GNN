import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
"""
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
"""
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features      # input size
        self.out_features = out_features    # output size
        # weight and bias are needed to learn
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # initialize the layer parameters
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # input will be infeatures*1 weight: infeatures*out
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # torch.mm(input, mat2, out=None) → Tensor
        # Performs a matrix multiplication of the matrices input and mat2.
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
