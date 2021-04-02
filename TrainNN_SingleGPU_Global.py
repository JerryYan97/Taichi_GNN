import time
import os, sys
import argparse
import torch.optim as optim
import torch
from src.Utils.utils_gcn import *
from src.NeuralNetworks.GlobalNN.GCN3D_Mar28_PoolingDeepGlobal import *
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import linalg as LA

os.makedirs('TrainedNN/GlobalNN', exist_ok=True)

case_id = 1011
cluster_num = 128

for root, dirs, files in os.walk("../runs/"):
    for name in files:
        os.remove(os.path.join(root, name))

writer = SummaryWriter('../runs/GCN_Global_' + str(case_id) + '_single')

# Training settings
epoch_num = 300
batch_size = 1
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1345, help='Random seed.')
parser.add_argument('--epochs', type=int, default=epoch_num, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
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
simDataset, case_info, cluster_parent, cluster_belong = load_global_data(case_id, cluster_num, "/SimData/TrainingData")
load_data_t_end = time.time()
print("data load time:", load_data_t_end - load_data_t_start)

dim = case_info['dim']

# Used to determine whether we are using a cluster.
pin_memory_option = False
if os.cpu_count() > 16:
    pin_memory_option = True

train_loader = DataLoader(dataset=simDataset, batch_size=batch_size,
                          shuffle=True,
                          num_workers=os.cpu_count(), pin_memory=pin_memory_option)

model = GCN3D_Mar28_PoolingDeepGlobal(
    nfeat=simDataset.input_features_num,
    graph_node_num=simDataset.node_num,
    cluster_num=simDataset.cluster_num,
    fc_out=dim,
    dropout=0,
    device=device,
    batch_num=batch_size
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
        metric1 = 0.0
        small_cnt = 0
        for data in train_loader:  # Iterate in batches over the training dataset.
            g_feat, o_feat = model(data.x.float().to(device),
                                   data.edge_index.to(device),
                                   data.batch.to(device),
                                   data.cluster.to(device))
            output = o_feat.reshape(data.num_graphs * simDataset.node_num, -1)

            loss_train = mse(output, data.y.float().to(device))

            output_cpu = output.cpu().detach()

            top_vec = LA.norm(output_cpu - data.y, dim=1).numpy()
            bottom_vec = (LA.norm(data.y, dim=1)).cpu().detach().numpy()

            top_vec_b = np.take(top_vec, simDataset.boundary_pt_idx)
            bottom_vec_b = np.take(bottom_vec, simDataset.boundary_pt_idx)

            big_idx = np.where(bottom_vec_b > 1e-10)
            top_cull_vec = np.take(top_vec_b, big_idx)
            bottom_cull_vec = np.take(bottom_vec_b, big_idx)
            tmp = top_cull_vec / bottom_cull_vec
            if np.isinf(tmp).any():
                raise Exception('Contain Elements that are inf!')
            small_cnt += (len(top_vec_b) - len(big_idx[0]))
            metric1 += np.sum(tmp)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train.cpu().detach().numpy()

        metric1 /= (len(simDataset.boundary_pt_idx) * len(simDataset.raw_file_names) - small_cnt)

        writer.add_scalar('metric1', metric1, epoch)
        print("Epoch:", epoch + 1,
              "metric1: ", metric1,
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
    writer.close()
