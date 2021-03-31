import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar31_LocalLinear_RBN_Mid(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar31_LocalLinear_RBN_Mid, self).__init__()
        self.fc1 = nn.Linear(nfeat, 720)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(720, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.bn1 = nn.BatchNorm1d(num_features=2048)

        self.fc4 = nn.Linear(2048, 1680)
        self.fc5 = nn.Linear(1680, 1280)
        self.bn2 = nn.BatchNorm1d(num_features=1280)

        self.fc6 = nn.Linear(1280, 960)
        self.fc7 = nn.Linear(960, 640)
        self.fc8 = nn.Linear(640, 512)
        self.bn3 = nn.BatchNorm1d(num_features=512)

        self.fc9 = nn.Linear(512, 256)
        self.fc10 = nn.Linear(256, 128)
        self.fc11 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(num_features=64)

        self.fc12 = nn.Linear(64, 32)
        self.fc13 = nn.Linear(32, 16)
        self.fc14 = nn.Linear(16, 8)
        self.bn5 = nn.BatchNorm1d(num_features=8)

        self.fc15 = nn.Linear(8, 3)
        self.fc16 = nn.Linear(3, 3)
        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.fc2(y))
        y = self.ELU(self.bn1(self.fc3(y)))

        y = self.ELU(self.fc4(y))
        y = self.ELU(self.bn2(self.fc5(y)))

        y = self.ELU(self.fc6(y))
        y = self.ELU(self.fc7(y))
        y = self.ELU(self.bn3(self.fc8(y)))

        y = self.ELU(self.fc9(y))
        y = self.ELU(self.fc10(y))
        y = self.ELU(self.bn4(self.fc11(y)))

        y = self.ELU(self.fc12(y))
        y = self.ELU(self.fc13(y))
        y = self.ELU(self.bn5(self.fc14(y)))

        y = self.ELU(self.fc15(y))
        y = self.fc16(y)
        return y
