import time
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from src.Utils.utils_gcn import *
from src.NeuralNetworks.GCN_net import *
import math
from torch.utils.data import Dataset, DataLoader

############## TENSORBOARD ########################
import sys
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/GCNJiarui-60_10_100')  # default `log_dir` is "runs" - we'll be more specific here
###################################################

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1345, help='Random seed.')
# PN -> PD:
# parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
# PD -> PN:
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=2e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--input_n', type=int, default=10, help='input feature length.')
parser.add_argument('--output_n', type=int, default=2, help='output feature length.')

# get parameters and check the cuda
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dim = 2
# Load data
mesh, adj, edge_index, simdatasets = load_txt_data2(1)

# Load whole dataset with DataLoader
train_loader = DataLoader(dataset=simdatasets, batch_size=1, shuffle=True, num_workers=0)
# test loader from 1 to 50 frame
test_loader = DataLoader(dataset=simdatasets, batch_size=1, shuffle=False)

# convert to an iterator and look at one random sample
dataiter = iter(train_loader)
data = dataiter.next()
inp, outp = data
print(len(simdatasets))
# print(inp, outp)

# Dummy Training loop
num_epochs = 2
len_data = len(simdatasets)
total_samples = len(simdatasets)
n_iterations = math.ceil(total_samples / 1)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations, Run your training process
        if (i + 1) % 5 == 0:
            pass


# Model and optimizer
node_num = mesh.num_vertices
input_features = 14
output_features = dim
model = GCN(nfeat=input_features, nhid=args.hidden, nclass=output_features, dropout=args.dropout).to(device)
mse = nn.MSELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if args.cuda:
    adj = adj.to(device)
    edge_index = edge_index.to(device)
    mse = mse.to(device)

def Sim_train():
    t = time.time()
    for epoch in range(args.epochs):
        for i, (inputs, outs) in enumerate(train_loader):
            inputs = torch.reshape(inputs, (node_num, -1)).float()
            inputs = inputs.to(device)
            outs = torch.reshape(outs, (node_num, -1)).float()
            outs = outs.to(device)
            model.train()

            zero_torch = torch.from_numpy(np.zeros((node_num, 2))).float().to(device)
            output = model(inputs, edge_index)
            # out = torch.true_divide(output - outs, outs).float()
            loss_train = mse(output, outs)
            # loss_train = mse(out, zero_torch)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # Evaluate validation set performance separately, deactivates dropout during validation run.
            # if not args.fastmode:
            #     model.eval()
            #     output = model(inputs, edge_index)

            print("Epoch:", i+1,
                  "loss_train: ", loss_train.cpu().detach().numpy(),
                  # 'acc_train: {:.4f}'.format(acc_train.item()),
                  # "loss_val: ", loss_val.cpu().detach().numpy(),
                  # 'acc_val: {:.4f}'.format(acc_val),
                  "time: ", time.time() - t,
                  "s")
            if (i + 1) % 10 == 0:
                ############## TENSORBOARD ########################
                writer.add_scalar('training loss', loss_train, (epoch*len_data)+i)
                writer.close()
                ###################################################


# Train model
t_total = time.time()
Sim_train()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

################################## savehe network #################################
# Specify a path
if not os.path.exists("./TrainedNN/"):
    os.mkdir("./TrainedNN/")

PATH = "TrainedNN/state_dict_model_zero_loss_1k.pt"
# Save
torch.save(model.state_dict(), PATH)

################################## save the state dict #################################
# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# print()
# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

