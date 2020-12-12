import time
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from src.Utils.utils_gcn import *
# from src.NeuralNetworks.GCNCNN_net import *
from src.NeuralNetworks.GCN_net_Dec9 import *
import math
from torch_geometric.data import DataLoader

############## TENSORBOARD ########################
import sys
from torch.utils.tensorboard import SummaryWriter
for root, dirs, files in os.walk("runs/"):
    for name in files:
        os.remove(os.path.join(root, name))
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
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=2e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')

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

# Load whole dataset with DataLoader
simDataset = load_txt_data(1, "/Outputs")
train_loader = DataLoader(dataset=simDataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=False)
# train_loader = DataLoader(dataset=simDataset, batch_size=64, shuffle=True, num_workers=1)

# For the purpose of dataset validation:
# for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()

# Model and optimizer
# model = GCN_CNN(nfeat=simDataset.input_features_num,
#                 nhid=args.hidden,
#                 nnode=simDataset.node_num,
#                 gcnout=20,
#                 cnnout=simDataset.node_num * dim,
#                 dropout=args.dropout).to(device)
model = GCN_net_Dec9(
                nfeat=simDataset.input_features_num,
                graph_node_num=simDataset.node_num,
                cluster_num=simDataset.cluster_num,
                gcn_hid1=32,
                gcn_out1=48,
                gcn_hid2=98,
                gcn_out2=128,
                fc_hid=60,
                fc_out=2,
                dropout=args.dropout).to(device)
mse = nn.MSELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model.cuda()
mse.cuda()

def Sim_train():
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        for data in train_loader:  # Iterate in batches over the training dataset.
            output = model(data.x.float().to(device),
                           data.edge_index.to(device),
                           data.num_graphs,
                           data.batch.to(device),
                           data.cluster.to(device)).reshape(data.num_graphs * simDataset.node_num, -1)

            l1_loss = torch.zeros(1).to(device)
            reg = 1e-6
            with torch.enable_grad():
                for name, param in model.named_parameters():
                    if 'bias' not in name:
                        if 'GCN' in name:
                            l1_loss = l1_loss + (reg * torch.sum(torch.abs(param.to(device))))

            loss_train = mse(output, data.y.float().to(device)) + l1_loss

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            # NOTE: The scale of loss_train is different from our previous training.
            # Here, it represents the total loss added by 'batch_size' graphs.
            print("Epoch:", epoch + 1,
                  "loss_train: ", loss_train.cpu().detach().numpy(),
                  "time: ", time.time() - t,
                  "s")
            if (epoch + 1) % 10 == 0:
                ############## TENSORBOARD ########################
                writer.add_scalar('training loss', loss_train, (epoch * len(simDataset)) + epoch)
                writer.close()
                ##################################################


# Train model
t_total = time.time()
Sim_train()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

################################## save the network #################################
# Specify a path
if not os.path.exists("./TrainedNN/"):
    os.mkdir("./TrainedNN/")

PATH = "TrainedNN/state_dict_model_zero_loss_1k.pt"
# Save
torch.save(model.state_dict(), PATH)
