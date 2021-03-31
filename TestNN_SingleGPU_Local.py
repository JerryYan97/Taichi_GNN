import time
import os, sys
import argparse
from src.Utils.utils_gcn import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar3_Local import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar10_Local import *
# from src.NeuralNetworks.LocalNN.VertNN_Feb28_LocalLinear import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar21_Local import *
# from src.NeuralNetworks.LocalNN.VertNN_Mar21_Local_MoreShallow import *
from src.NeuralNetworks.LocalNN.VertNN_Mar31_Local_RBN_Mid import *
import pickle

from src.NeuralNetworks.GlobalNN.GCN3D_Mar28_PoolingDeepGlobal import *

import math
from torch_geometric.data import DataLoader

# Training settings
simulator_feature_num = 18
case_id = 1009
cluster_num = 256
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')

# get parameters and check the cuda
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read case_info (We cannot use PyMesh on the cluster)
case_info = pickle.load(open(os.getcwd() + "/MeshModels/MeshInfo/case_info" + str(case_id) + "_RHF.p", "rb"))

# Loading global NN:
# on_hash_table: old index (Not culled) -> New index (Culled);
# no_hash_table: New index (Culled) -> Old index (Not culled);
GLOBAL_NN_PATH = "TrainedNN/GlobalNN/GlobalNN_LowPolyArm_18.pt"
culled_cluster_num, graph_node_num, edge_idx, hash_table, culled_cluster, culled_idx = load_global_info(case_info, case_id, cluster_num)
global_model = GCN3D_Mar28_PoolingDeepGlobal(
    nfeat=simulator_feature_num,
    graph_node_num=graph_node_num,
    cluster_num=culled_cluster_num,
    fc_out=3,
    dropout=0,
    device=device,
    batch_num=1  # Global Batch size should always be 1.
).to(device)
global_model.load_state_dict(torch.load(GLOBAL_NN_PATH))

# Loading local NN:
LOCAL_NN_PATH = "TrainedNN/LocalNN/LocalNN_LowPolyArm18.pt"

# Model and optimizer
simDataset = load_local_data(case_info, hash_table, edge_idx, culled_idx, culled_cluster,
                             simulator_feature_num, global_model.global_feat_num, culled_cluster_num,
                             global_model, device, 0, "/SimData/TestingData", True)
dim = case_info['dim']
test_loader = DataLoader(dataset=simDataset, batch_size=simDataset.boundary_node_num, shuffle=False)

local_model = VertNN_Mar31_LocalLinear_RBN_Mid(
    nfeat=simDataset.input_features_num,
    fc_out=simDataset.output_features_num,
    dropout=0,
    device=device
).to(device)

local_model.load_state_dict(torch.load(LOCAL_NN_PATH))


def RunNN():
    local_model.eval()
    metric1 = 0.0
    small_cnt = 0
    with torch.no_grad():
        i = 0
        for data in test_loader:
            current_file_names = data['filename']
            unique_name = set(current_file_names)
            if len(unique_name) != 1:
                raise Exception('A batch should only contain 1 frame.')
            outname = "SimData/RunNNRes/test_res_" + current_file_names[0]
            output = local_model(data['x'].float().to(device))
            output_cpu = output.cpu().detach()
            npinputs = data['x'].cpu().detach().numpy()
            npouts = output_cpu.numpy()

            # Calculate metric1
            top_vec = torch_LA.norm(output_cpu - data['y'], dim=1).numpy()
            bottom_vec = (torch_LA.norm(data['y'], dim=1)).cpu().detach().numpy()

            big_idx = np.where(bottom_vec > 1e-10)
            top_cull_vec = np.take(top_vec, big_idx)
            bottom_cull_vec = np.take(bottom_vec, big_idx)
            tmp = top_cull_vec / bottom_cull_vec
            if np.isinf(tmp).any():
                raise Exception('Contain Elements that are inf!')
            small_cnt += (len(top_vec) - len(big_idx[0]))
            metric1 += np.sum(tmp)

            file_name_len = len(current_file_names[0])
            file_id_str = current_file_names[0][file_name_len - 9:file_name_len - 4]
            file_id = int(file_id_str)

            print("File id:", file_id)
            # PD displacement, PD-GNN, PN displacement
            dis = npinputs[:, 0:dim]
            outfinal = np.hstack((dis, npouts))
            outfinal = np.hstack((outfinal, data['y']))
            np.savetxt(outname, outfinal, delimiter=',')
            i += 1
        metric1 /= (len(simDataset)-small_cnt)
        print("metric1:", metric1)


if __name__ == '__main__':
    os.makedirs('SimData/RunNNRes/', exist_ok=True)
    for root, dirs, files in os.walk("SimData/RunNNRes/"):
        for name in files:
            os.remove(os.path.join(root, name))
    # Save a boundary pts idx file (Boundary pts idx is for the culled data)
    vis_info = {"local_bd_idx": simDataset.boundary_node_mesh_idx.numpy()}
    pickle.dump(vis_info, open("SimData/RunNNRes/vis_info_" + str(case_id) + ".p", "wb"))
    RunNN()
