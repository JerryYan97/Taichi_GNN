import time
import os, sys
import argparse
import torch.optim as optim
from src.Utils.utils_gcn import *
from src.NeuralNetworks.GlobalNN.GCN3D_Feb16_PoolingDeepGlobal import *
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.makedirs('TrainedNN/GlobalNN', exist_ok=True)

for root, dirs, files in os.walk("../runs/"):
    for name in files:
        os.remove(os.path.join(root, name))

writer = SummaryWriter('../runs/GCN_Global_1009_single')  # default `log_dir` is "runs" - we'll be more specific here
###################################################

# Training settings
epoch_num = 500
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1345, help='Random seed.')
parser.add_argument('--epochs', type=int, default=epoch_num, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.003, help='Initial learning rate.')
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
simDataset, case_info, cluster_parent, cluster_belong = load_data(1009, 256, "/SimData/TrainingData")
load_data_t_end = time.time()
print("data load time:", load_data_t_end - load_data_t_start)

dim = case_info['dim']

train_loader = DataLoader(dataset=simDataset, batch_size=1, shuffle=True, num_workers=16, pin_memory=False)

model = GCN3D_Feb16_PoolingDeepGlobal(
    nfeat=simDataset.input_features_num,
    graph_node_num=simDataset.node_num,
    cluster_num=simDataset.cluster_num,
    fc_out=dim,
    dropout=0,
    device=device
)

model.to(device)
mse = nn.MSELoss(reduction='sum').to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6, verbose=True, eps=1e-20)

model.cuda()
mse.cuda()
bs = train_loader.batch_size


def Sim_train():
    record_times = 0
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for data in train_loader:  # Iterate in batches over the training dataset.
            output = model(data.x.float().to(device),
                           data.edge_index.to(device),
                           data.num_graphs,
                           data.batch.to(device),
                           data.cluster.to(device)).reshape(data.num_graphs * simDataset.node_num, -1)

            loss_train = mse(output, data.y.float().to(device))     # + l1_loss
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train.cpu().detach().numpy()

        writer.add_scalar('training loss', epoch_loss / simDataset.len(), epoch)
        writer.close()
        print("Epoch:", epoch + 1,
              "avg frame training loss: ", epoch_loss / simDataset.len(),
              "time: ", time.time() - t, "s")
        scheduler.step(epoch_loss)

        # record the model
        if epoch > epoch_num-80:
            if epoch % 4 == 0:
                torch.save(model.state_dict(),
                           "TrainedNN/GlobalNN/GlobalNN_" + case_info['case_name'] + "_" + str(record_times) + ".pt")
                record_times = record_times + 1


if __name__ == '__main__':
    t_total = time.time()
    Sim_train()
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
