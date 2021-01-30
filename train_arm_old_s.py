import time
import os, sys
import argparse, math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from src.Utils.utils_gcn import *
from src.NeuralNetworks.GCN3D import *
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

for root, dirs, files in os.walk("../runs/"):
    for name in files:
        os.remove(os.path.join(root, name))
writer = SummaryWriter('../runs/GCN_1007_single')  # default `log_dir` is "runs" - we'll be more specific here
###################################################
torch.cuda.empty_cache()
print("clean up the cuda cache! ")
torch.multiprocessing.set_sharing_strategy('file_system')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1345, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=2e-3, help='Weight decay (L2 loss on parameters).')

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


load_data_t_start = time.time()
simDataset, case_info, cluster_parent, cluster_belong = load_data(1007, 512, "/SimData/TrainingData")
# print("ok!")
load_data_t_end = time.time()
print("data load time:", load_data_t_end - load_data_t_start)

dim = case_info['dim']

train_loader = DataLoader(dataset=simDataset, batch_size=1, shuffle=True, num_workers=16, pin_memory=False)

h1 = 256
h2 = 512
h3 = 512
model = GCNOLD(nfeat=simDataset.input_features_num,
               graph_node_num=simDataset.node_num,
               cluster_num=simDataset.cluster_num,
               gcn_hid1=h1,
               gcn_hid2=h2,
               gcn_hid3=h3,
               gcn_out1=512,
               gcn_out2=256,
               gcn_out3=256,
               fc_hid=16,
               fc_out=dim,
               num_gcns=3,
               num_linear=4)

# model = torch.nn.DataParallel(model)
model.to(device)
mse = nn.MSELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)  #, weight_decay=args.weight_decay)

model.cuda()
mse.cuda()
bs = train_loader.batch_size


def Sim_train():
    start_record = False
    record_times = 0
    t = time.time()
    model.train()
    this_loss = 0.0
    last_loss = 0.0
    Up = torch.zeros([simDataset.node_num * bs, 256], dtype=torch.float32)
    ii = 0
    iii = 1
    for epoch in range(args.epochs):
        record_flag = False
        if math.fabs((this_loss - last_loss) / iii) < 0.0001 or epoch > 40:
            start_record = True
        last_loss = this_loss
        this_loss = 0.0
        iii = 0
        for data in train_loader:  # Iterate in batches over the training dataset.
            output = model(data.x.float().to(device),
                           data.edge_index.to(device),
                           data.num_graphs,
                           data.batch.to(device),
                           data.cluster.to(device),
                           Up.to(device),
                           cluster_parent,
                           cluster_belong).reshape(data.num_graphs * simDataset.node_num, -1)

            l1_loss = torch.zeros(1).to(device)
            reg = 1e-4
            with torch.enable_grad():
                for name, param in model.named_parameters():
                    if 'bias' not in name:
                        if 'GCN1' in name or 'GCN2' in name or 'GCN3' in name or 'GCN4' in name or 'GCN5' in name:
                            l1_loss = l1_loss + (reg * torch.sum(torch.abs(param.to(device))))

            loss_train = mse(output, data.y.float().to(device)) + l1_loss
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            this_loss += loss_train.cpu().detach().numpy()

            if ii % 40 == 0:
                print("Epoch:", epoch + 1,
                      "loss_train: ", loss_train.cpu().detach().numpy(),
                      "time: ", time.time() - t, "s")
                if not record_flag:
                    writer.add_scalar('training loss', loss_train, ii)
                    writer.close()
                    record_flag = True

            ii = ii + 1
            iii = iii + 1

        # record the model
        if start_record is True:
            if epoch % 4 == 0:
                torch.save(model.state_dict(), "TrainedNN/Arm_" + str(record_times) + ".pt")
                record_times = record_times + 1


if __name__ == '__main__':
    Sim_train()     # Train model
    t_total = time.time()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
