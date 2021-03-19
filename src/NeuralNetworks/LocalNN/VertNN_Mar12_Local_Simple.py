import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar12_LocalLinear_Simple(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar12_LocalLinear_Simple, self).__init__()
        self.fc1 = nn.Linear(nfeat, 1024)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(num_features=256)

        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 8)

        self.fc7 = nn.Linear(8, 3)
        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.fc2(y))
        y = self.ELU(self.bn1(self.fc3(y)))

        y = self.ELU(self.fc4(y))
        y = self.ELU(self.fc5(y))
        y = self.ELU(self.fc6(y))

        y = self.fc7(y)

        return y
