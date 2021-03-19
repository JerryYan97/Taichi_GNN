import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar12_LocalLinear(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar12_LocalLinear, self).__init__()
        self.fc1 = nn.Linear(nfeat, 1024)  # Hidden layers' width is influenced by your cluster num.
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.bn2 = nn.BatchNorm1d(num_features=2048)

        self.fc3 = nn.Linear(2048, 3200)
        self.bn3 = nn.BatchNorm1d(num_features=3200)
        self.fc4 = nn.Linear(3200, 3000)
        self.bn4 = nn.BatchNorm1d(num_features=3000)

        self.fc5 = nn.Linear(3000, 2800)
        self.bn5 = nn.BatchNorm1d(num_features=2800)
        self.fc6 = nn.Linear(2800, 2560)
        self.bn6 = nn.BatchNorm1d(num_features=2560)

        self.fc7 = nn.Linear(2560, 2000)
        self.bn7 = nn.BatchNorm1d(num_features=2000)
        self.fc8 = nn.Linear(2000, 1280)
        self.bn8 = nn.BatchNorm1d(num_features=1280)

        self.fc9 = nn.Linear(1280, 960)
        self.bn9 = nn.BatchNorm1d(num_features=960)
        self.fc10 = nn.Linear(960, 640)
        self.bn10 = nn.BatchNorm1d(num_features=640)

        self.fc11 = nn.Linear(640, 512)
        self.bn11 = nn.BatchNorm1d(num_features=512)
        self.fc12 = nn.Linear(512, 256)
        self.bn12 = nn.BatchNorm1d(num_features=256)

        self.fc13 = nn.Linear(256, 128)
        self.bn13 = nn.BatchNorm1d(num_features=128)
        self.fc14 = nn.Linear(128, 64)
        self.bn14 = nn.BatchNorm1d(num_features=64)

        self.fc15 = nn.Linear(64, 32)
        self.bn15 = nn.BatchNorm1d(num_features=32)
        self.fc16 = nn.Linear(32, 16)
        self.bn16 = nn.BatchNorm1d(num_features=16)

        self.fc17 = nn.Linear(16, 8)
        self.bn17 = nn.BatchNorm1d(num_features=8)
        self.fc18 = nn.Linear(8, 3)
        self.bn18 = nn.BatchNorm1d(num_features=3)

        self.fc19 = nn.Linear(3, 3)
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

        y = self.ELU(self.bn9(self.fc9(y)))
        y = self.ELU(self.bn10(self.fc10(y)))
        y = self.ELU(self.bn11(self.fc11(y)))
        y = self.ELU(self.bn12(self.fc12(y)))

        y = self.ELU(self.bn13(self.fc13(y)))
        y = self.ELU(self.bn14(self.fc14(y)))
        y = self.ELU(self.bn15(self.fc15(y)))
        y = self.ELU(self.bn16(self.fc16(y)))

        y = self.ELU(self.bn17(self.fc17(y)))
        y = self.ELU(self.bn18(self.fc18(y)))

        y = self.fc19(y)

        return y
