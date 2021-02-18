import time
import os, sys
import argparse, math
import numpy as np
import torch
import torch.optim as optim
from src.Utils.utils_gcn import *
from src.NeuralNetworks.LocalNN.VertNN_Feb16_LocalLinear import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.makedirs('TrainedNN/LocalNN', exist_ok=True)

for root, dirs, files in os.walk("../runs/"):
    for name in files:
        os.remove(os.path.join(root, name))

writer = SummaryWriter('../runs/GCN_Local_1009_single')
###################################################

# Training settings
epoch_num = 500
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1345, help='Random seed.')
parser.add_argument('--epochs', type=int, default=epoch_num, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
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
load_data_t_end = time.time()
print("data load time:", load_data_t_end - load_data_t_start)

dim = case_info['dim']

train_loader = DataLoader(dataset=simDataset, batch_size=512, shuffle=True, num_workers=16)

model = VertNN_Feb16_LocalLinear(
    nfeat=simDataset.input_features_num,
    fc_out=simDataset.output_features_num,
    dropout=0,
    device=device
).to(device)

mse = nn.MSELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6, verbose=True, eps=1e-20)


def Sim_train():
    record_times = 0
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i_batch, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(sample_batched['x'].float().to(device))
            loss_train = mse(output, sample_batched['y'].float().to(device))
            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train.cpu().detach().numpy()

        writer.add_scalar('Local avg frame training loss', epoch_loss / simDataset.len(), epoch)
        writer.close()
        print("Epoch:", epoch + 1,
              "avg frame training loss: ", epoch_loss / simDataset.len(),
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
    Sim_train()
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
