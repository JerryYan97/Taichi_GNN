import torch
import torch.nn as nn


# Local vertex NN
class VertNN_Mar10_LocalLinear_Wider(nn.Module):
    def __init__(self, nfeat, fc_out, dropout, device):
        super(VertNN_Mar10_LocalLinear_Wider, self).__init__()
        self.fc1 = nn.Linear(nfeat, 1024)  # Hidden layers' width is influenced by your cluster num.
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.bn2 = nn.BatchNorm1d(num_features=2048)

        self.fc3 = nn.Linear(2048, 3200)
        self.bn3 = nn.BatchNorm1d(num_features=3200)
        self.fc4 = nn.Linear(3200, 2400)
        self.bn4 = nn.BatchNorm1d(num_features=2400)

        self.fc5 = nn.Linear(2400, 1210)
        self.bn5 = nn.BatchNorm1d(num_features=1210)
        self.fc6 = nn.Linear(1210, 512)
        self.bn6 = nn.BatchNorm1d(num_features=512)

        self.fc7 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(num_features=256)
        self.fc8 = nn.Linear(256, 128)
        self.bn8 = nn.BatchNorm1d(num_features=128)

        self.fc9 = nn.Linear(128, 64)
        self.bn9 = nn.BatchNorm1d(num_features=64)
        self.fc10 = nn.Linear(64, 9)
        self.bn10 = nn.BatchNorm1d(num_features=9)

        self.fc11 = nn.Linear(9, 3)
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
        y = self.fc11(y)
        return y
