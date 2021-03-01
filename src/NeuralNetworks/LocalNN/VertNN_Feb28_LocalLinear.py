import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Feb28_LocalLinear(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Feb28_LocalLinear, self).__init__()
        self.fc1 = nn.Linear(nfeat, 128)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, fc_out)
        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.fc2(y))
        y = self.ELU(self.fc3(y))
        y = self.ELU(self.fc4(y))
        y = self.fc5(y)
        return y
