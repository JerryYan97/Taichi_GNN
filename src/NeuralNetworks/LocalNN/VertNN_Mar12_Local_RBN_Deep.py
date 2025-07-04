import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar12_LocalLinear_RBN_Deep(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar12_LocalLinear_RBN_Deep, self).__init__()
        self.fc1 = nn.Linear(nfeat, 720)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(720, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 3200)
        self.bn1 = nn.BatchNorm1d(num_features=3200)

        self.fc5 = nn.Linear(3200, 3000)
        self.fc6 = nn.Linear(3000, 2800)
        self.fc7 = nn.Linear(2800, 2560)
        self.fc8 = nn.Linear(2560, 2400)
        self.bn2 = nn.BatchNorm1d(num_features=2400)

        self.fc9 = nn.Linear(2400, 2200)
        self.fc10 = nn.Linear(2200, 1980)
        self.fc11 = nn.Linear(1980, 1280)
        self.bn3 = nn.BatchNorm1d(num_features=1280)

        self.fc12 = nn.Linear(1280, 960)
        self.fc13 = nn.Linear(960, 800)
        self.fc14 = nn.Linear(800, 640)
        self.bn4 = nn.BatchNorm1d(num_features=640)

        self.fc15 = nn.Linear(640, 512)
        self.fc16 = nn.Linear(512, 256)
        self.fc17 = nn.Linear(256, 128)
        self.fc18 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(num_features=64)

        self.fc19 = nn.Linear(64, 32)
        self.fc20 = nn.Linear(32, 16)
        self.fc21 = nn.Linear(16, 8)
        self.bn6 = nn.BatchNorm1d(num_features=8)

        self.fc22 = nn.Linear(8, 3)
        self.fc23 = nn.Linear(3, 3)
        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.fc2(y))
        y = self.ELU(self.fc3(y))
        y = self.ELU(self.bn1(self.fc4(y)))

        y = self.ELU(self.fc5(y))
        y = self.ELU(self.fc6(y))
        y = self.ELU(self.fc7(y))
        y = self.ELU(self.bn2(self.fc8(y)))

        y = self.ELU(self.fc9(y))
        y = self.ELU(self.fc10(y))
        y = self.ELU(self.bn3(self.fc11(y)))

        y = self.ELU(self.fc12(y))
        y = self.ELU(self.fc13(y))
        y = self.ELU(self.bn4(self.fc14(y)))

        y = self.ELU(self.fc15(y))
        y = self.ELU(self.fc16(y))
        y = self.ELU(self.fc17(y))
        y = self.ELU(self.bn5(self.fc18(y)))

        y = self.ELU(self.fc19(y))
        y = self.ELU(self.fc20(y))
        y = self.ELU(self.bn6(self.fc21(y)))

        y = self.ELU(self.fc22(y))
        y = self.fc23(y)
        return y
