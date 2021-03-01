import time
import os, sys
import argparse
from src.Utils.utils_gcn import *
from src.NeuralNetworks.LocalNN.VertNN_Feb28_LocalLinear import *

import math
from torch_geometric.data import DataLoader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')

# get parameters and check the cuda
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify a path
PATH = "TrainedNN/LocalNN/LocalNN_LowPolyArm18.pt"

# Model and optimizer
simDataset, case_info = load_local_data(1009, 256, "/SimData/TestingData")  # load test data
dim = case_info['dim']
test_loader = DataLoader(dataset=simDataset, batch_size=simDataset.boundary_node_num, shuffle=False)

model = VertNN_Feb28_LocalLinear(
    nfeat=simDataset.input_features_num,
    fc_out=simDataset.output_features_num,
    dropout=0,
    device=device
).to(device)

model.load_state_dict(torch.load(PATH))
mse = nn.MSELoss(reduction='sum').to(device)


def RunNN():
    model.eval()
    overall_loss = 0.0
    with torch.no_grad():
        i = 0
        for data in test_loader:
            current_file_names = data['filename']
            unique_name = set(current_file_names)
            if len(unique_name) != 1:
                raise Exception('A batch should only contain 1 frame.')
            outname = "SimData/RunNNRes/test_res_" + current_file_names[0]
            output = model(data['x'].float().to(device))
            npinputs = data['x'].cpu().detach().numpy()
            npouts = output.cpu().detach().numpy()
            loss = mse(output, data['y'].float().to(device))
            overall_loss += loss

            file_name_len = len(current_file_names[0])
            file_id_str = current_file_names[0][file_name_len - 9:file_name_len - 4]
            file_id = int(file_id_str)

            print("File id:", file_id,
                  "Frame MSE loss: ", loss.cpu().detach().numpy())
            # PD displacement, PD-GNN, PN displacement
            dis = npinputs[:, 0:dim]
            outfinal = np.hstack((dis, npouts))
            outfinal = np.hstack((outfinal, data['y']))
            np.savetxt(outname, outfinal, delimiter=',')
            i += 1
        print("overall_avg_loss:", overall_loss / float(i))


if __name__ == '__main__':
    os.makedirs('SimData/RunNNRes/', exist_ok=True)
    for root, dirs, files in os.walk("SimData/RunNNRes/"):
        for name in files:
            os.remove(os.path.join(root, name))
    # Save a boundary pts idx file
    b_pts_idx_name = "SimData/RunNNRes/b_pts_idx_" + case_info['case_name'] + ".csv"
    np.savetxt(b_pts_idx_name, simDataset.boundary_node_mesh_idx, delimiter=',', fmt='%d')
    RunNN()
