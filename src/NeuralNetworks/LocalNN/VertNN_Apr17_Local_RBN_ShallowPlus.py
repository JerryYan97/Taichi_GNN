import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Apr17_LocalLinear_RBN_ShallowPlus(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Apr17_LocalLinear_RBN_ShallowPlus, self).__init__()
        self.fc1 = nn.Linear(nfeat, 64)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(num_features=32)

        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 3)
        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.bn1(self.fc2(y)))
        y = self.ELU(self.fc3(y))
        y = self.fc4(y)
        return y
