from utils_gcn import *
from GCN_net import *

loss = nn.MSELoss(reduction='sum')

for i in range(10):
    input = torch.randn(2, requires_grad=True)
    target = torch.randn(2)
    print("input:\n", input, "output:\n", target)
    output = loss(input, target)
    # output.backward()
    print("loss: ", output.cpu().detach().numpy())

