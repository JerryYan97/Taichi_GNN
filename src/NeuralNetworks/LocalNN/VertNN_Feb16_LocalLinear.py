import torch
import torch.nn as nn
import torch.nn.functional as F


# Local vertex NN
class VertNN_Feb16_LocalLinear(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Feb16_LocalLinear, self).__init__()
        self.fc1 = nn.Linear(nfeat, 8)
        self.fc2 = nn.Linear(8, fc_out)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        return y
