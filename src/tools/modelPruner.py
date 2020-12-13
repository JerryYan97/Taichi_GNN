import torch
from torch import nn
import os, sys
from torch_geometric.nn import GCNConv
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from NeuralNetworks.GCN_net_Dec9 import*

# test class
model = GCN_net_Dec9(nfeat=14, graph_node_num=231,
                     gcn_hid1=32, gcn_out=48,
                     unet_hid=8, unet_out=16,
                     fc_hid=60, fc_out=2, dropout=0.3)

def showParams(model):
    for name, param in model.named_parameters():
        print(name, ", shape: ", param.shape)
        if name.find('weight') != -1 and name.find('GCN') != -1:
            print("weight: ", param)

def weightPrune(model, threshold, model_name):
    old_params = {}
    for name, params in model.named_parameters():
        old_params[name] = params.clone()

    for name, param in model.named_parameters():
        if name.find('weight') != -1 and name.find('GCN') != -1:
            print("weight: ", param)
            print("weight ", name, " ", param.shape)
            r_times = 0
            for i in range(old_params[name].shape[0]):
                for j in range(old_params[name].shape[1]):
                    if abs(old_params[name][i][j]) < threshold:
                        old_params[name][i][j] = 0.0
                        r_times = r_times + 1
            print("after revising", r_times, "times weight: ", param)

    for name, params in model.named_parameters():
        params.data.copy_(old_params[name])

    if not os.path.exists("./TrainedNN/"):  # Specify a path
        os.mkdir("./TrainedNN/")
    PATH = model_name[:model_name.find('.pt')] + "_prune.pt"
    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    PATH = "../../TrainedNN/state_dict_model_zero_loss_1k_prune.pt"
    print(sys.path)
    model.load_state_dict(torch.load(PATH))
    showParams(model)
    weightPrune(model, 0.1, PATH)
