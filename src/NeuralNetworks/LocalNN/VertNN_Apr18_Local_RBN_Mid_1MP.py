import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Apr18_LocalLinear_RBN_Mid_1MP(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Apr18_LocalLinear_RBN_Mid_1MP, self).__init__()
        self.fc1 = nn.Linear(nfeat, 720)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(720, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 3200)
        self.bn1 = nn.BatchNorm1d(num_features=3200)

        self.fc5 = nn.Linear(3200, 3000)
        self.mp1 = nn.MaxPool1d(2)
        self.fc6 = nn.Linear(1500, 1280)
        self.fc7 = nn.Linear(1280, 960)
        self.bn2 = nn.BatchNorm1d(num_features=960)

        self.fc8 = nn.Linear(960, 720)
        self.fc9 = nn.Linear(720, 240)
        self.fc10 = nn.Linear(240, 128)
        self.bn3 = nn.BatchNorm1d(num_features=128)

        self.fc11 = nn.Linear(128, 64)
        self.fc12 = nn.Linear(64, 32)
        self.fc13 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(num_features=16)

        self.fc14 = nn.Linear(16, 8)
        self.fc15 = nn.Linear(8, 3)

        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.fc2(y))
        y = self.ELU(self.fc3(y))
        y = self.ELU(self.bn1(self.fc4(y)))

        y = self.ELU(self.fc5(y))
        y = y.unsqueeze(1)
        y = self.mp1(y)
        y = y.squeeze(1)
        y = self.ELU(self.fc6(y))
        y = self.ELU(self.bn2(self.fc7(y)))

        y = self.ELU(self.fc8(y))
        y = self.ELU(self.fc9(y))
        y = self.ELU(self.bn3(self.fc10(y)))

        y = self.ELU(self.fc11(y))
        y = self.ELU(self.fc12(y))
        y = self.ELU(self.bn4(self.fc13(y)))

        y = self.ELU(self.fc14(y))
        y = self.fc15(y)
        return y
