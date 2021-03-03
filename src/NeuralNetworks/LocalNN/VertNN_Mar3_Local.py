import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar3_LocalLinear(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar3_LocalLinear, self).__init__()
        self.fc1 = nn.Linear(nfeat, 512)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(512, 480)
        self.fc3 = nn.Linear(480, 360)
        self.fc4 = nn.Linear(360, 320)

        self.fc5 = nn.Linear(320, 300)
        self.fc6 = nn.Linear(300, 280)
        self.fc7 = nn.Linear(280, 260)
        self.fc8 = nn.Linear(260, 256)

        self.fc9 = nn.Linear(256, 240)
        self.fc10 = nn.Linear(240, 220)
        self.fc11 = nn.Linear(220, 196)
        self.fc12 = nn.Linear(196, 128)

        self.fc13 = nn.Linear(128, 121)
        self.fc14 = nn.Linear(121, 110)
        self.fc15 = nn.Linear(110, 100)
        self.fc16 = nn.Linear(100, 96)

        self.fc17 = nn.Linear(96, 81)
        self.fc18 = nn.Linear(81, 72)
        self.fc19 = nn.Linear(72, 56)
        self.fc20 = nn.Linear(56, 48)

        self.fc21 = nn.Linear(48, 40)
        self.fc22 = nn.Linear(40, 32)
        self.fc23 = nn.Linear(32, 28)
        self.fc24 = nn.Linear(28, 24)

        self.fc25 = nn.Linear(24, 16)
        self.fc26 = nn.Linear(16, 8)
        self.fc27 = nn.Linear(8, 4)
        self.fc28 = nn.Linear(4, fc_out)
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
        y = self.ELU(self.fc12(y))

        y = self.ELU(self.fc13(y))
        y = self.ELU(self.fc14(y))
        y = self.ELU(self.fc15(y))
        y = self.ELU(self.fc16(y))

        y = self.ELU(self.fc17(y))
        y = self.ELU(self.fc18(y))
        y = self.ELU(self.fc19(y))
        y = self.ELU(self.fc20(y))

        y = self.ELU(self.fc21(y))
        y = self.ELU(self.fc22(y))
        y = self.ELU(self.fc23(y))
        y = self.ELU(self.fc24(y))

        y = self.ELU(self.fc25(y))
        y = self.ELU(self.fc26(y))
        y = self.ELU(self.fc27(y))
        y = self.fc28(y)
        return y
