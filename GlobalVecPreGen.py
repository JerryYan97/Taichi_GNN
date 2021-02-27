import argparse
from src.Utils.utils_gcn import *
from src.NeuralNetworks.GlobalNN.GCN3D_Feb16_PoolingDeepGlobal import *
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')

# get parameters and check the cuda
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('../runs/GlobalVecPreGen')
PATH = "TrainedNN/GlobalNN/GlobalNN_IrregularBeam_18.pt"

# Model and optimizer
simDataset, case_info, cluster_parent, cluster_belong = load_data(1011, 8, "/SimData/TrainingData")  # load test data
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
        for data in test_loader:
            outname = "SimData/PreGenGlobalFeatureVec/gvec_" + simDataset.raw_file_names[i]
            output, g_vec = model(data.x.float().to(device),
                                  data.edge_index.to(device),
                                  data.num_graphs,
                                  data.batch.to(device),
                                  data.cluster.to(device))
            output = output.reshape(data.num_graphs * simDataset.node_num, -1)
            loss = mse(output, data.y.float().to(device))
            writer.add_scalar('frame -- gvec loss', loss, i)
            print("Frame:", i,
                  "loss: ", loss.cpu().detach().numpy())
            np.savetxt(outname, g_vec.cpu().detach().numpy(), delimiter=',')
            i = i + 1


if __name__ == '__main__':
    for root, dirs, files in os.walk("../runs/"):
        for name in files:
            os.remove(os.path.join(root, name))
    os.makedirs('SimData/PreGenGlobalFeatureVec/', exist_ok=True)
    for root, dirs, files in os.walk("SimData/PreGenGlobalFeatureVec/"):
        for name in files:
            os.remove(os.path.join(root, name))
    RunNN()
    writer.close()
