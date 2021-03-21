import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar21_LocalLinear(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar21_LocalLinear, self).__init__()
        self.fc1 = nn.Linear(nfeat, 1024)  # Hidden layers' width is influenced by your cluster num.
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 960)
        self.bn2 = nn.BatchNorm1d(num_features=960)

        self.fc3 = nn.Linear(960, 720)
        self.bn3 = nn.BatchNorm1d(num_features=720)
        self.fc4 = nn.Linear(720, 640)
        self.bn4 = nn.BatchNorm1d(num_features=640)

        self.fc5 = nn.Linear(640, 512)
        self.bn5 = nn.BatchNorm1d(num_features=512)
        self.fc6 = nn.Linear(512, 480)
        self.bn6 = nn.BatchNorm1d(num_features=480)

        self.fc7 = nn.Linear(480, 320)
        self.bn7 = nn.BatchNorm1d(num_features=320)
        self.fc8 = nn.Linear(320, 256)
        self.bn8 = nn.BatchNorm1d(num_features=256)

        self.fc9 = nn.Linear(256, 128)
        self.bn9 = nn.BatchNorm1d(num_features=128)
        self.fc10 = nn.Linear(128, 96)
        self.bn10 = nn.BatchNorm1d(num_features=96)

        self.fc11 = nn.Linear(96, 72)
        self.bn11 = nn.BatchNorm1d(num_features=72)
        self.fc12 = nn.Linear(72, 64)
        self.bn12 = nn.BatchNorm1d(num_features=64)

        self.fc13 = nn.Linear(64, 32)
        self.bn13 = nn.BatchNorm1d(num_features=32)
        self.fc14 = nn.Linear(32, 16)
        self.bn14 = nn.BatchNorm1d(num_features=16)

        self.fc15 = nn.Linear(16, 8)
        self.bn15 = nn.BatchNorm1d(num_features=8)
        self.fc16 = nn.Linear(8, 3)

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

        y = self.fc16(y)

        return y
