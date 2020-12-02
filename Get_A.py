import time
import os, sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from src.Utils.utils_gcn import *
from src.NeuralNetworks.GCN_net import *
from src.NeuralNetworks.GCNCNN_net import *
import math
from torch.utils.data import DataLoader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=546, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--input_n', type=int, default=10, help='input feature length.')
parser.add_argument('--output_n', type=int, default=2, help='output feature length.')

# get parameters and check the cuda
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify a path
PATH = "TrainedNN/state_dict_model_zero_loss_1k.pt"
dim = 2

# Model and optimizer
mesh, edge_index, simdatasets = load_st_txt_data(1)  # load test data
# test loader from 1 to 50 frame
test_loader = DataLoader(dataset=simdatasets, batch_size=1, shuffle=False)

node_num = mesh.num_vertices
input_features = 14
output_features = dim
# model = GCN(nfeat=input_features, nhid=args.hidden, nclass=output_features, dropout=args.dropout).to(device)
model = GCN(nfeat=input_features, nhid=args.hidden, nclass=output_features, gcnout=20, cnnout=node_num*dim,  dropout=args.dropout).to(device)

# Load
model.load_state_dict(torch.load(PATH))
mse = nn.MSELoss(reduction='sum').to(device)

if args.cuda:
    edge_index = edge_index.to(device)
    mse = mse.to(device)

def AAA():
    model.eval()
    with torch.no_grad():
        for i, (inputs, outs) in enumerate(test_loader):
            ii = str(i).zfill(2)
            outname = "TestResult/frame" + ii + ".csv"
            inputs = torch.reshape(inputs, (node_num, -1)).float()
            inputs = inputs.to(device)
            outs = torch.reshape(outs, (node_num, -1)).float()
            outs = outs.to(device)

            output = model(inputs, edge_index)
            output = torch.reshape(output, (node_num, -1))

            npinputs = inputs.cpu().detach().numpy()
            npouts = output.cpu().detach().numpy()

            l1_loss = torch.zeros(1).to(device)
            reg = 1e-6
            with torch.enable_grad():
                for name, param in model.named_parameters():
                    if 'bias' not in name:
                        if 'GCN' in name:
                            l1_loss = l1_loss + (reg * torch.sum(torch.abs(param.to(device))))

            loss = mse(output, outs) + l1_loss
            # loss = mse(output, outs)
            print("Frame:", i,
                  "loss: ", loss.cpu().detach().numpy())
            # print("outs:\n", outs.cpu().detach().numpy())
            # print("output:\n", output.cpu().detach().numpy())

            # PD displacement, PD-GNN, ???, PN displacement
            dis = npinputs[:, 0:2]
            after_add = np.add(dis, npouts)
            outfinal = np.hstack((dis, npouts))
            outfinal = np.hstack((outfinal, after_add))
            outfinal = np.hstack((outfinal, outs.cpu().detach().numpy()))
            np.savetxt(outname, outfinal, delimiter=',')


if __name__ == '__main__':
    if not os.path.exists("TestResult"):
        os.makedirs("TestResult")
    for root, dirs, files in os.walk("TestResult/"):
        for name in files:
            os.remove(os.path.join(root, name))
    AAA()
