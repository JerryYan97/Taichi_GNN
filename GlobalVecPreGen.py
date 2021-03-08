import argparse
from src.Utils.utils_gcn import *
from src.NeuralNetworks.GlobalNN.GCN3D_Feb16_PoolingDeepGlobal import *
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import linalg as LA

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')

# get parameters and check the cuda
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('../runs/GlobalVecPreGen')
PATH = "TrainedNN/GlobalNN/GlobalNN_LowPolyArm_18.pt"

# Model and optimizer
simDataset, case_info, cluster_parent, cluster_belong = load_data(1009, 8, "/SimData/TestingData")
dim = case_info['dim']
test_loader = DataLoader(dataset=simDataset, batch_size=1, shuffle=False)

model = GCN3D_Feb16_PoolingDeepGlobal(
    graph_node_num=simDataset.node_num,
    cluster_num=simDataset.cluster_num,
    nfeat=simDataset.input_features_num,
    fc_out=dim,
    dropout=0,
    device=device
).to(device)

model.load_state_dict(torch.load(PATH))
mse = nn.MSELoss(reduction='sum').to(device)
node_num = simDataset.node_num


def RunNN():
    model.eval()
    with torch.no_grad():
        i = 0
        metric1_total = 0.0
        total_cnt = 0
        for data in test_loader:
            metric1_frame = 0.0
            outname = "SimData/PreGenGlobalFeatureVec/gvec_" + simDataset.raw_file_names[i]
            outname2 = "SimData/PreGenSpannedGlobalFeatureVec/gvec_full_" + simDataset.raw_file_names[i]
            output, g_vec = model(data.x.float().to(device),
                                  data.edge_index.to(device),
                                  data.num_graphs,
                                  data.batch.to(device),
                                  data.cluster.to(device))
            output = output.reshape(data.num_graphs * simDataset.node_num, -1)
            output_cpu = output.cpu().detach()
            # loss = mse(output, data.y.float().to(device))

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
            total_cnt += len(big_idx[0])
            metric1_frame += np.sum(tmp)
            metric1_total += metric1_frame
            metric1_frame /= len(big_idx[0])

            writer.add_scalar('metric 1 for a frame: ', metric1_frame, i)
            print("Frame:", i,
                  "metric 1 for this frame: ", metric1_frame)
            np.savetxt(outname, g_vec.cpu().detach().numpy(), delimiter=',')
            np.savetxt(outname2, output.cpu().detach().numpy(), delimiter=',')
            i = i + 1

        # Print the final metric1 loss
        print("Avg metric 1 val: ", metric1_total / total_cnt)


if __name__ == '__main__':
    for root, dirs, files in os.walk("../runs/"):
        for name in files:
            os.remove(os.path.join(root, name))
    os.makedirs('SimData/PreGenGlobalFeatureVec/', exist_ok=True)
    for root, dirs, files in os.walk("SimData/PreGenGlobalFeatureVec/"):
        for name in files:
            os.remove(os.path.join(root, name))
    os.makedirs('SimData/PreGenSpannedGlobalFeatureVec/', exist_ok=True)
    for root, dirs, files in os.walk("SimData/PreGenSpannedGlobalFeatureVec/"):
        for name in files:
            os.remove(os.path.join(root, name))
    RunNN()
    writer.close()
