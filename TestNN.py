import time
import os, sys
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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=546, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

# get parameters and check the cuda
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify a path
PATH = "TrainedNN/state_dict_model_zero_loss_1k.pt"
# PATH = "TrainedNN/state_dict_model_zero_loss_1k_prune.pt"

# Model and optimizer
simDataset, case_info = load_data(1007, "/SimData/TestingData")  # load test data
dim = case_info['dim']
test_loader = DataLoader(dataset=simDataset, batch_size=1, shuffle=False)
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
                fc_out=dim,
                dropout=args.dropout).to(device)
model.load_state_dict(torch.load(PATH))
mse = nn.MSELoss(reduction='sum').to(device)
node_num = simDataset.node_num


def RunNN():
    model.eval()
    with torch.no_grad():
        i = 0
        for data in test_loader:
            ii = str(i).zfill(5)
            outname = "SimData/RunNNRes/frame" + ii + ".csv"
            output = model(data.x.float().to(device),
                           data.edge_index.to(device),
                           data.num_graphs,
                           data.batch.to(device),
                           data.cluster.to(device)).reshape(data.num_graphs * simDataset.node_num, -1)
            npinputs = data.x.cpu().detach().numpy()
            npouts = output.cpu().detach().numpy()
            l1_loss = torch.zeros(1).to(device)
            reg = 1e-6
            with torch.enable_grad():
                for name, param in model.named_parameters():
                    if 'bias' not in name:
                        if 'GCN' in name:
                            l1_loss = l1_loss + (reg * torch.sum(torch.abs(param.to(device))))

            loss = mse(output, data.y.float().to(device)) + l1_loss
            print("Frame:", i,
                  "loss: ", loss.cpu().detach().numpy())
            # PD displacement, PD-GNN, ???, PN displacement
            dis = npinputs[:, 0:dim]
            after_add = np.add(dis, npouts)
            outfinal = np.hstack((dis, npouts))
            outfinal = np.hstack((outfinal, after_add))
            outfinal = np.hstack((outfinal, data.y))
            np.savetxt(outname, outfinal, delimiter=',')
            i = i + 1


if __name__ == '__main__':
    os.makedirs('SimData/RunNNRes/', exist_ok=True)
    for root, dirs, files in os.walk("SimData/RunNNRes/"):
        for name in files:
            os.remove(os.path.join(root, name))
    RunNN()
