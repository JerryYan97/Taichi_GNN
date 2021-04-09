import time
import os, sys
import argparse, math
import numpy as np
import torch
import torch.optim as optim
from src.Utils.utils_gcn import *
# from src.NeuralNetworks.LocalNN.VertNN_Feb28_LocalLinear import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar3_Local import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar21_Local_MoreShallow import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar12_Local_Simple import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar12_Local import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar12_Local_ReduceBN import *
from src.NeuralNetworks.LocalNN.VertNN_Mar12_Local_RBN_Deep import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar31_Local_RBN_Mid import *

from src.NeuralNetworks.GlobalNN.GCN3D_Mar28_PoolingDeepGlobal import *

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import linalg as LA

os.makedirs('TrainedNN/LocalNN', exist_ok=True)

for root, dirs, files in os.walk("./runs/"):
    for name in files:
        os.remove(os.path.join(root, name))

writer = SummaryWriter('./runs/GCN_Local_1009_single')
###################################################

# Training settings
epoch_num = 300
simulator_feature_num = 18
case_id = 1011
cluster_num = 128
additional_note = '7sets_data'

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1345, help='Random seed.')
parser.add_argument('--epochs', type=int, default=epoch_num, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')

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

# Read case_info (We cannot use PyMesh on the cluster)
train_info = pickle.load(open(os.getcwd() + "/SimData/TrainingDataPickle/train_info" + str(case_id) + "_" +
                              str(cluster_num) + "_" + additional_note + ".p", "rb"))

# Load and set global NN:
GLOBAL_NN_PATH = "TrainedNN/GlobalNN/GlobalNN_IrregularBeam_18.pt"
global_model = GCN3D_Mar28_PoolingDeepGlobal(
    nfeat=simulator_feature_num,
    graph_node_num=train_info['graph_node_num'],
    cluster_num=train_info['culled_cluster_num'],
    fc_out=3,
    dropout=0,
    device=device,
    batch_num=1  # Global Batch size should always be 1.
).to(device)
global_model.load_state_dict(torch.load(GLOBAL_NN_PATH))

load_data_t_start = time.time()
simDataset = load_local_data(train_info, simulator_feature_num, global_model.global_feat_num,
                             global_model, device, True)

load_data_t_end = time.time()
print("data load time:", load_data_t_end - load_data_t_start)

train_loader = DataLoader(dataset=simDataset,
                          batch_size=256,
                          shuffle=True,
                          num_workers=os.cpu_count(),
                          pin_memory=True)

# model = VertNN_Feb28_LocalLinear(
#     nfeat=simDataset.input_features_num,
#     fc_out=simDataset.output_features_num,
#     dropout=0,
#     device=device
# ).to(device)

local_model = VertNN_Mar12_LocalLinear_RBN_Deep(
    nfeat=simDataset.input_features_num,
    fc_out=simDataset.output_features_num,
    dropout=0,
    device=device
).to(device)


mse = nn.MSELoss().to(device)
optimizer = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True, eps=1e-20)


def Sim_train():
    record_times = 0
    t = time.time()
    local_model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        # metric1: Per point average of the norm(PDNN - (PN-PD))/norm((PN-PD))
        # metric2: Max norm(PDNN - (PN-PD))/norm((PN-PD)) in one epcho
        metric1 = 0.0
        metric2 = 0.0
        small_cnt = 0
        for i_batch, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()
            output = local_model(sample_batched['x'].float().to(device))
            loss_train = mse(output, sample_batched['y'].float().to(device))

            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train.cpu().detach().numpy()
            output_cpu = output.cpu().detach()
            top_vec = LA.norm(output_cpu - sample_batched['y'], dim=1).numpy()
            bottom_vec = (LA.norm(sample_batched['y'], dim=1)).cpu().detach().numpy()

            big_idx = np.where(bottom_vec > 1e-10)
            top_cull_vec = np.take(top_vec, big_idx)
            bottom_cull_vec = np.take(bottom_vec, big_idx)
            tmp = top_cull_vec / bottom_cull_vec
            if np.isinf(tmp).any():
                raise Exception('Contain Elements that are inf!')
            small_cnt += (len(top_vec) - len(big_idx[0]))
            metric1 += np.sum(tmp)
            if metric2 < tmp.max():
                metric2 = tmp.max()

        # print("top vec sum:", top_vec_square_sum, "\n epcho MSE loss:", epoch_loss)
        metric1 /= (len(simDataset)-small_cnt)
        epoch_loss /= len(simDataset)
        writer.add_scalar('Metric1(AVG relative pt displacement)', metric1, epoch)
        writer.add_scalar('Metric2(MAX relative pt displacement)', metric2, epoch)
        writer.add_scalar('Avg pt MSELoss', epoch_loss, epoch)
        writer.close()
        print("Epoch:", epoch + 1,
              "metric1: ", metric1,
              "metric2: ", metric2,
              "Avg pt MSELoss: ", epoch_loss,
              "small cnt:", small_cnt,
              "boundary pt cnt:", simDataset.boundary_node_num,
              "time: ", time.time() - t, "s")
        scheduler.step(epoch_loss)

        # record the model
        if epoch > epoch_num-80:
            if epoch % 4 == 0:
                torch.save(local_model.state_dict(),
                           "TrainedNN/LocalNN/LocalNN_" + train_info['case_name'] + str(record_times) + ".pt")
                record_times = record_times + 1


if __name__ == '__main__':
    t_total = time.time()
    Sim_train()
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
