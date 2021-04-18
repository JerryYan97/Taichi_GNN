import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Apr18_LocalLinear_RBN_Shallow_1MP(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Apr18_LocalLinear_RBN_Shallow_1MP, self).__init__()
        self.fc1 = nn.Linear(nfeat, 720)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(720, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.bn1 = nn.BatchNorm1d(num_features=2048)

        self.fc5 = nn.Linear(2048, 1024)
        self.mp1 = nn.MaxPool1d(4)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(num_features=64)

        self.fc8 = nn.Linear(64, 8)
        self.fc9 = nn.Linear(8, 3)

        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.fc2(y))
        y = self.ELU(self.bn1(self.fc3(y)))

        y = self.ELU(self.fc5(y))
        y = y.unsqueeze(1)
        y = self.mp1(y)
        y = y.squeeze(1)
        y = self.ELU(self.fc6(y))
        y = self.ELU(self.bn2(self.fc7(y)))

        y = self.ELU(self.fc8(y))
        y = self.fc9(y)

        return y
