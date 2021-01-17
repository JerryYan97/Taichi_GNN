import time
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim as optim
import torch.nn as nn
from src.Utils.utils_gcn import *
from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from src.NeuralNetworks.GCNCNN_net import *
# from src.NeuralNetworks.GCN_net_Dec9 import *
# from src.NeuralNetworks.GCN3D_Jan14 import *
from src.NeuralNetworks.GCN3D_Jan15 import *
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
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
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

# Load whole dataset with DataLoader
# Optimization record:
# case 1001 -- 9.8G:
# t1: 0.0037581920623779297  t2: 0.03488945960998535  t3: 0.000118255615234375  t4: 0.0011484622955322266
# t5-1: 0.016495704650878906  t5-2: 0.004931211471557617  t5-3: 184.7452096939087
# t5: 186.16088032722473
# After opt:
# ~50s in total
load_data_t_start = time.time()
simDataset, case_info = load_data(1008, "/SimData/TrainingData", transform=T.Compose([T.NormalizeFeatures(),
                                                                                      T.ToSparseTensor()]))
load_data_t_end = time.time()
print("data load time:", load_data_t_end - load_data_t_start)

dim = case_info['dim']

train_loader = DataLoader(dataset=simDataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=False)
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
# model = GCN_net_Dec9(
# model = GCN3D_Jan14(
#                 nfeat=simDataset.input_features_num,
#                 graph_node_num=simDataset.node_num,
#                 cluster_num=simDataset.cluster_num,
#                 gcn_hid1=32 * 2,
#                 gcn_out1=48 * 2,
#                 gcn_hid2=98 * 2,
#                 gcn_out2=128 * 2,
#                 fc_hid=60 * 2,
#                 fc_out=dim,
#                 dropout=args.dropout).to(device)

model = GCN3D_Jan15(num_layers=64,
                    alpha=0.1,
                    theta=0.5,
                    graph_node_num=simDataset.node_num,
                    nfeat=simDataset.input_features_num,
                    fc_out=dim,
                    dropout=args.dropout).to(device)

mse = nn.MSELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model.cuda()
mse.cuda()

def Sim_train():
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        record_flag = False
        for data in train_loader:  # Iterate in batches over the training dataset.
            output = model(data.x.float().to(device),
                           # data.edge_index.to(device),
                           gcn_norm(data.adj_t).to(device),
                           data.num_graphs,
                           data.batch.to(device),
                           data.cluster.to(device)).reshape(data.num_graphs * simDataset.node_num, -1)

            l1_loss = torch.zeros(1).to(device)
            reg = 1e-4
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
            if not record_flag:
                ############## TENSORBOARD ########################
                writer.add_scalar('training loss', loss_train, (epoch * len(simDataset)) + epoch)
                writer.close()
                record_flag = True
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
