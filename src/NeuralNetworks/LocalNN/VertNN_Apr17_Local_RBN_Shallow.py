import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Apr17_LocalLinear_RBN_Shallow(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Apr17_LocalLinear_RBN_Shallow, self).__init__()
        self.fc1 = nn.Linear(nfeat, 720)  # Hidden layers' width is influenced by your cluster num.
        self.fc2 = nn.Linear(720, 1280)
        self.bn1 = nn.BatchNorm1d(num_features=1280)

        self.fc3 = nn.Linear(1280, 960)
        self.fc4 = nn.Linear(960, 512)
        self.bn2 = nn.BatchNorm1d(num_features=512)

        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(num_features=128)

        self.fc7 = nn.Linear(128, 32)
        self.fc8 = nn.Linear(32, 8)

        self.fc9 = nn.Linear(8, 3)
        self.ELU = nn.ELU()

    def forward(self, x):
        y = self.ELU(self.fc1(x))
        y = self.ELU(self.bn1(self.fc2(y)))

        y = self.ELU(self.fc3(y))
        y = self.ELU(self.bn2(self.fc4(y)))

        y = self.ELU(self.fc5(y))
        y = self.ELU(self.bn3(self.fc6(y)))

        y = self.ELU(self.fc7(y))
        y = self.ELU(self.fc8(y))

        y = self.fc9(y)
        return y
