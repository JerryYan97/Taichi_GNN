import time
import os, sys
import argparse, math
import numpy as np
import torch
import torch.optim as optim
from src.Utils.utils_gcn import *
# from src.NeuralNetworks.LocalNN.VertNN_Feb28_LocalLinear import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar2_Local import *
from src.NeuralNetworks.LocalNN.VertNN_Mar3_Local import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import linalg as LA

os.makedirs('TrainedNN/LocalNN', exist_ok=True)

for root, dirs, files in os.walk("../runs/"):
    for name in files:
        os.remove(os.path.join(root, name))

writer = SummaryWriter('../runs/GCN_Local_1009_single')
###################################################

# Training settings
epoch_num = 300
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1345, help='Random seed.')
parser.add_argument('--epochs', type=int, default=epoch_num, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
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


load_data_t_start = time.time()
simDataset, case_info = load_local_data(1009, 256, "/SimData/TrainingData")
simDataset.to_device(device)
load_data_t_end = time.time()
print("data load time:", load_data_t_end - load_data_t_start)

dim = case_info['dim']

train_loader = DataLoader(dataset=simDataset,
                          batch_size=256,
                          shuffle=True,
                          num_workers=0,
                          # num_workers=os.cpu_count(),
                          pin_memory=False)

# model = VertNN_Feb28_LocalLinear(
#     nfeat=simDataset.input_features_num,
#     fc_out=simDataset.output_features_num,
#     dropout=0,
#     device=device
# ).to(device)

model = VertNN_Mar3_LocalLinear(
    nfeat=simDataset.input_features_num,
    fc_out=simDataset.output_features_num,
    dropout=0,
    device=device
).to(device)

# mse = nn.MSELoss(reduction='sum').to(device)
mse = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True, eps=1e-20)


def Sim_train():
    record_times = 0
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        # metric1: Per point average of the norm(PDNN - (PN-PD))/norm((PN-PD))
        # metric2: Max norm(PDNN - (PN-PD))/norm((PN-PD)) in one epcho
        metric1 = 0.0
        metric2 = 0.0
        # zero_cnt = 0
        small_cnt = 0
        # top_vec_square_sum = 0.0
        # data_itr = iter(train_loader)
        # data = data_itr.next()
        for i_batch, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()
            # output = model(sample_batched['x'].float().to(device))
            # loss_train = mse(output, sample_batched['y'].float().to(device))
            output = model(sample_batched['x'])
            loss_train = mse(output, sample_batched['y'])
            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train.cpu().detach().numpy()
            # output_cpu = output.cpu().detach()
            # top_vec = LA.norm(output_cpu - sample_batched['y'], dim=1).numpy()
            top_vec = LA.norm(output - sample_batched['y'], dim=1).cpu().detach().numpy()
            bottom_vec = (LA.norm(sample_batched['y'], dim=1)).cpu().detach().numpy()
            # nonzero_idx = np.where(bottom_vec != 0.0)
            big_idx = np.where(bottom_vec > 1e-10)
            top_cull_vec = np.take(top_vec, big_idx)
            bottom_cull_vec = np.take(bottom_vec, big_idx)
            # small_idx = np.where(bottom_cull_vec < 0.00001)
            # small_cnt += len(small_idx[0])
            tmp = top_cull_vec / bottom_cull_vec
            if np.isinf(tmp).any():
                raise Exception('Contain Elements that are inf!')
            # zero_cnt += (len(top_vec) - len(nonzero_idx[0]))
            small_cnt += (len(top_vec) - len(big_idx[0]))
            metric1 += np.sum(tmp)
            # top_vec_square_sum += np.sum(np.square(top_vec))
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
                torch.save(model.state_dict(),
                           "TrainedNN/LocalNN/LocalNN_" + case_info['case_name'] + str(record_times) + ".pt")
                record_times = record_times + 1


if __name__ == '__main__':
    t_total = time.time()
    # data_itr = iter(train_loader)
    # data = data_itr.next()
    Sim_train()
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
