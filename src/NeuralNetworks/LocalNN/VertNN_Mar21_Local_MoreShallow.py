import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar21_LocalLinear_MoreShallow(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar21_LocalLinear_MoreShallow, self).__init__()
        self.fc1 = nn.Linear(nfeat, 1024)  # Hidden layers' width is influenced by your cluster num.
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 720)
        self.bn2 = nn.BatchNorm1d(num_features=720)

        self.fc3 = nn.Linear(720, 320)
        self.bn3 = nn.BatchNorm1d(num_features=320)
        self.fc4 = nn.Linear(320, 128)
        self.bn4 = nn.BatchNorm1d(num_features=128)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(num_features=64)
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(num_features=32)

        self.fc7 = nn.Linear(32, 16)
        self.bn7 = nn.BatchNorm1d(num_features=16)
        self.fc8 = nn.Linear(16, 8)
        self.bn8 = nn.BatchNorm1d(num_features=8)

        self.fc9 = nn.Linear(8, 3)

        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.bn1(self.fc1(x)))
        y = self.ELU(self.bn2(self.fc2(y)))
        y = self.ELU(self.bn3(self.fc3(y)))
        y = self.ELU(self.bn4(self.fc4(y)))

        y = self.ELU(self.bn5(self.fc5(y)))
        y = self.ELU(self.bn6(self.fc6(y)))
        y = self.ELU(self.bn7(self.fc7(y)))
        y = self.ELU(self.bn8(self.fc8(y)))

        y = self.fc9(y)

        return y
