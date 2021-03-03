import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar3_LocalLinear(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar3_LocalLinear, self).__init__()
        self.fc1 = nn.Linear(nfeat, 512)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(512, 480)
        self.fc3 = nn.Linear(480, 320)
        self.fc4 = nn.Linear(320, 256)

        self.bn1 = nn.BatchNorm1d(num_features=256)

        self.fc5 = nn.Linear(256, 196)
        self.fc6 = nn.Linear(196, 156)
        self.fc7 = nn.Linear(156, 140)
        self.fc8 = nn.Linear(140, 128)

        self.fc9 = nn.Linear(128, 121)
        self.fc10 = nn.Linear(121, 110)
        self.fc11 = nn.Linear(110, 100)
        self.fc12 = nn.Linear(100, 96)

        self.bn2 = nn.BatchNorm1d(num_features=96)

        self.fc13 = nn.Linear(96, 128)
        self.fc14 = nn.Linear(128, 72)
        self.fc15 = nn.Linear(72, 56)
        self.fc16 = nn.Linear(56, 48)

        self.fc17 = nn.Linear(48, 32)
        self.fc18 = nn.Linear(32, 16)
        self.fc19 = nn.Linear(16, 8)
        self.fc20 = nn.Linear(8, fc_out)

        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.fc2(y))
        y = self.ELU(self.fc3(y))
        y = self.ELU(self.bn1(self.fc4(y)))

        y = self.ELU(self.fc5(y))
        y = self.ELU(self.fc6(y))
        y = self.ELU(self.fc7(y))
        y = self.ELU(self.fc8(y))

        y = self.ELU(self.fc9(y))
        y = self.ELU(self.fc10(y))
        y = self.ELU(self.fc11(y))
        y = self.ELU(self.bn2(self.fc12(y)))

        y = self.ELU(self.fc13(y))
        y = self.ELU(self.fc14(y))
        y = self.ELU(self.fc15(y))
        y = self.ELU(self.fc16(y))

        y = self.ELU(self.fc17(y))
        y = self.ELU(self.fc18(y))
        y = self.ELU(self.fc19(y))
        y = self.fc20(y)

        return y
