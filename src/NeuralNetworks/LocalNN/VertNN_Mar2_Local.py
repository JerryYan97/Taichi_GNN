import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar2_LocalLinear(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar2_LocalLinear, self).__init__()
        self.fc1 = nn.Linear(nfeat, 256)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 96)
        self.fc4 = nn.Linear(96, 64)

        self.fc5 = nn.Linear(64, 56)
        self.fc6 = nn.Linear(56, 49)
        self.fc7 = nn.Linear(49, 32)
        self.fc8 = nn.Linear(32, 24)

        self.fc9 = nn.Linear(24, 16)
        self.fc10 = nn.Linear(16, 8)
        self.fc11 = nn.Linear(8, 4)
        self.fc12 = nn.Linear(4, fc_out)
        # self.fc1 = nn.Linear(nfeat, 32)  # Hidden layers' width is influenced by your cluster num.
        # self.fc2 = nn.Linear(32, 64)
        # self.fc3 = nn.Linear(64, 24)
        # self.fc4 = nn.Linear(24, 4)
        # self.fc5 = nn.Linear(4, fc_out)
        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.fc2(y))
        y = self.ELU(self.fc3(y))
        y = self.ELU(self.fc4(y))

        y = self.ELU(self.fc5(y))
        y = self.ELU(self.fc6(y))
        y = self.ELU(self.fc7(y))
        y = self.ELU(self.fc8(y))

        y = self.ELU(self.fc9(y))
        y = self.ELU(self.fc10(y))
        y = self.ELU(self.fc11(y))
        y = self.fc12(y)
        return y
