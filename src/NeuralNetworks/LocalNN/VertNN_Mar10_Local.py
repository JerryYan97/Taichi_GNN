import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar10_LocalLinear(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar10_LocalLinear, self).__init__()
        self.fc1 = nn.Linear(nfeat, 1024)  # Hidden layers' width is influenced by your cluster num.
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 960)
        self.bn2 = nn.BatchNorm1d(num_features=960)

        self.fc3 = nn.Linear(960, 860)
        self.bn3 = nn.BatchNorm1d(num_features=860)
        self.fc4 = nn.Linear(860, 720)
        self.bn4 = nn.BatchNorm1d(num_features=720)

        self.fc5 = nn.Linear(720, 640)
        self.bn5 = nn.BatchNorm1d(num_features=640)
        self.fc6 = nn.Linear(640, 580)
        self.bn6 = nn.BatchNorm1d(num_features=580)

        self.fc7 = nn.Linear(580, 512)
        self.bn7 = nn.BatchNorm1d(num_features=512)
        self.fc8 = nn.Linear(512, 480)
        self.bn8 = nn.BatchNorm1d(num_features=480)

        self.fc9 = nn.Linear(480, 420)
        self.bn9 = nn.BatchNorm1d(num_features=420)
        self.fc10 = nn.Linear(420, 400)
        self.bn10 = nn.BatchNorm1d(num_features=400)

        self.fc11 = nn.Linear(400, 320)
        self.bn11 = nn.BatchNorm1d(num_features=320)
        self.fc12 = nn.Linear(320, 280)
        self.bn12 = nn.BatchNorm1d(num_features=280)

        self.fc13 = nn.Linear(280, 256)
        self.bn13 = nn.BatchNorm1d(num_features=256)
        self.fc14 = nn.Linear(256, 200)
        self.bn14 = nn.BatchNorm1d(num_features=200)

        self.fc15 = nn.Linear(200, 156)
        self.bn15 = nn.BatchNorm1d(num_features=156)
        self.fc16 = nn.Linear(156, 128)
        self.bn16 = nn.BatchNorm1d(num_features=128)

        self.fc17 = nn.Linear(128, 96)
        self.bn17 = nn.BatchNorm1d(num_features=96)
        self.fc18 = nn.Linear(96, 80)
        self.bn18 = nn.BatchNorm1d(num_features=80)

        self.fc19 = nn.Linear(80, 72)
        self.bn19 = nn.BatchNorm1d(num_features=72)
        self.fc20 = nn.Linear(72, 64)
        self.bn20 = nn.BatchNorm1d(num_features=64)

        self.fc21 = nn.Linear(64, 32)
        self.bn21 = nn.BatchNorm1d(num_features=32)
        self.fc22 = nn.Linear(32, 16)
        self.bn22 = nn.BatchNorm1d(num_features=16)

        self.fc23 = nn.Linear(16, 8)
        self.bn23 = nn.BatchNorm1d(num_features=8)
        self.fc24 = nn.Linear(8, 6)
        self.bn24 = nn.BatchNorm1d(num_features=6)

        self.fc25 = nn.Linear(6, 3)

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
        y = self.ELU(self.bn19(self.fc19(y)))
        y = self.ELU(self.bn20(self.fc20(y)))

        y = self.ELU(self.bn21(self.fc21(y)))
        y = self.ELU(self.bn22(self.fc22(y)))
        y = self.ELU(self.bn23(self.fc23(y)))
        y = self.ELU(self.bn24(self.fc24(y)))

        y = self.fc25(y)

        return y
