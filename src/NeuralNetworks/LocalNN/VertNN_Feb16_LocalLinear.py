import torch
import torch.nn as nn
import torch.nn.functional as F


# Local vertex NN
class VertNN_Feb16_LocalLinear(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Feb16_LocalLinear, self).__init__()
        self.fc1 = nn.Linear(nfeat, 128)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, fc_out)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y
