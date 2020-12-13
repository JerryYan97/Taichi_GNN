import torch
from torch import nn
from torch_geometric.nn import GCNConv

# test class
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.GCN1 = GCNConv(12, 8)
        self.GCN3 = GCNConv(8, 4)  # middle layers
        self.GCN5 = GCNConv(4, 2)  # output channel is a int number#
        self.fc1 = nn.Linear(231 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, adj, num_graphs):
        x = self.GCN1(x, adj)
        x = torch.tanh(x)
        x = self.GCN3(x, adj)
        x = torch.tanh(x)
        x = self.GCN5(x, adj)
        x = x.view(231*2)
        x = self.fc1(x)
        x = nn.ELU(x)
        x = self.fc2(x)
        x = nn.ELU(x)
        x = self.fc3(x)
        return x

model = LeNet()

def showParams(model):
    for name, param in model.named_parameters():
        print(name, ", shape: ", param.shape)

def weightPrune(model, threshold):
    for name, param in model.named_parameters():
        if name.find('weight') != -1 and name.find('GCN') != -1:
            print("weight: ", param)
            print("weight ", name, " ", param.shape)
            r_times = 0
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    if abs(param[i][j]) < threshold:
                        param[i][j] = 0.0
                        r_times = r_times + 1
            print("after revising", r_times, "times weight: ", param)


if __name__ == '__main__':
    showParams(model)
    weightPrune(model, 0.1)
