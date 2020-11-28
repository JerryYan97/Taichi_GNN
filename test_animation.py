import time
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from utils_gcn import *
from GCN_net import *
import math
from torch.utils.data import Dataset, DataLoader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=546, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0008, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=600, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--input_n', type=int, default=10, help='input feature length.')
parser.add_argument('--output_n', type=int, default=2, help='output feature length.')

############## TENSORBOARD ########################
import sys
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/TEST')  # default `log_dir` is "runs" - we'll be more specific here

################################## savehe network #################################
# Specify a path
PATH = "state_dict_model_2.pt"
dim = 2

# Model and optimizer
mesh, edge_index, simdatasets = load_st_txt_data(1)  # load test data
# test loader from 1 to 50 frame
test_loader = DataLoader(dataset=simdatasets, batch_size=1, shuffle=False)

node_num = mesh.num_vertices
input_features = 8
output_features = node_num * dim
model = GCN(nfeat=input_features, nhid=args.hidden, nclass=output_features, dropout=args.dropout).to(device)

# Load
model.load_state_dict(torch.load(PATH))
model.eval()
