import numpy as np
import scipy.sparse as sp
import torch
from .reader import *
import os
from collections import defaultdict
from torch_geometric.data import InMemoryDataset, Data


class SIM_Data_Geo(InMemoryDataset):
    def __init__(self, filepath, mesh_edge_idx, i_features_num, o_features_num, node_num, transform=None,
                 pre_transform=None):
        super(SIM_Data_Geo, self).__init__(None, transform, pre_transform)
        self._files = []
        for _, _, files in os.walk(filepath):
            self._files.extend(files)
        self._files.sort()
        self._edge_idx = mesh_edge_idx
        self._filepath = filepath
        self._input_features_num = i_features_num
        self._node_num = node_num
        self._output_features_num = o_features_num

        sample_list = []
        for idx in range(self.len()):
            fperframe = np.genfromtxt(self._filepath + "/" + self._files[idx], delimiter=',')
            other = fperframe[:, 4:]
            pn_dis = fperframe[:, 2:4]
            pd_dis = fperframe[:, 0:2]  # a[start:stop] items start through stop-1
            y_data = torch.from_numpy(np.subtract(pn_dis, pd_dis).reshape((self.node_num, -1)))
            x_data = torch.from_numpy(np.hstack((pd_dis, other)).reshape((self.node_num, -1)))
            sample = Data(x=x_data, edge_index=self._edge_idx, y=y_data)
            if self.transform:
                sample = self.transform(sample)
            sample_list.append(sample)

        self.data, self.slices = self.collate(sample_list)

    @property
    def raw_file_names(self):
        return self._files

    @property
    def input_features_num(self):
        return self._input_features_num

    @property
    def output_features_num(self):
        return self._output_features_num

    @property
    def node_num(self):
        return self._node_num

    def len(self):
        return len(self.raw_file_names)


def load_txt_data(objpath, path="/Outputs"):
    file_dir = os.getcwd()
    file_dir = file_dir + path

    mesh, _, _, _ = read(int(objpath))
    edges = set()
    for [i, j, k] in mesh.faces:
        edges.add((i, j))
        edges.add((j, k))
        edges.add((k, i))
    edge_index = np.zeros(shape=(2, 0), dtype=np.int32)
    for [i, j, k] in mesh.faces:
        if (j, i) not in edges:
            edge_index = np.hstack((edge_index, [[j], [i]]))
            edge_index = np.hstack((edge_index, [[i], [j]]))
        if (k, j) not in edges:
            edge_index = np.hstack((edge_index, [[j], [k]]))
            edge_index = np.hstack((edge_index, [[k], [j]]))
        if (i, k) not in edges:
            edge_index = np.hstack((edge_index, [[k], [i]]))
            edge_index = np.hstack((edge_index, [[i], [k]]))
    edge_index = torch.LongTensor(edge_index)
    dataset = SIM_Data_Geo(file_dir, edge_index, 14, 2, mesh.num_vertices)
    return dataset


# def load_st_txt_data(objpath, path="Outputs_T"):
#     """Load citation network dataset (cora only for now)"""
#     file_dir = os.getcwd()
#     file_dir = file_dir + "/Outputs_T"
#     print('\nWith Tensor Transform')
#     dataset = SIM_Data(file_dir, transform=ToTensor())
#
#     first_data = dataset[0]
#     features, labels = first_data
#     # print(type(features), type(labels))
#     print(features.shape, "          ",  labels.shape)
#     # print(features, labels)
#
#     mesh, _, _, _ = read(int(objpath))  # build graph
#     edges = set()
#     for [i, j, k] in mesh.faces:
#         edges.add((i, j))
#         edges.add((j, k))
#         edges.add((k, i))
#     boundary_points_ = set()
#     boundary_edges_ = np.zeros(shape=(0, 2), dtype=np.int32)
#     edge_index = np.zeros(shape=(2, 0), dtype=np.int32)
#     for [i, j, k] in mesh.faces:
#         if (j, i) not in edges:
#             boundary_points_.update([j, i])
#             boundary_edges_ = np.vstack((boundary_edges_, [j, i]))
#             edge_index = np.hstack((edge_index, [[j], [i]]))
#             edge_index = np.hstack((edge_index, [[i], [j]]))
#         if (k, j) not in edges:
#             boundary_points_.update([k, j])
#             boundary_edges_ = np.vstack((boundary_edges_, [k, j]))
#             edge_index = np.hstack((edge_index, [[j], [k]]))
#             edge_index = np.hstack((edge_index, [[k], [j]]))
#         if (i, k) not in edges:
#             boundary_points_.update([i, k])
#             boundary_edges_ = np.vstack((boundary_edges_, [i, k]))
#             edge_index = np.hstack((edge_index, [[k], [i]]))
#             edge_index = np.hstack((edge_index, [[i], [k]]))
#
#     edge_index = torch.LongTensor(edge_index)
#
#     return mesh, edge_index, dataset


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    errors = np.subtract(output, labels)
    sum_errors = np.sum(errors)
    return sum_errors


# definition of function
def generate_edges(graph):
    edges = []                                  # for each node in graph
    for node in graph:
        for neighbour in graph[node]:           # for each neighbour node of a single node
            edges.append((node, neighbour))     # if edge exists then append
    return edges


def addEdge(graph, u, v):
    graph[u].append(v)


# definition of function
def buildGraph(mesh, edge):
    graph = defaultdict(list)       # function for adding edge to graph
    for e in edge:                  # declaration of graph as dictionary
        addEdge(graph, e[0], e[1])
    print(generate_edges(graph))    # Driver Function call， to print generated graph
    return graph


if __name__ == "__main__":
    # mesh, _, _, _ = read(int(1))  # build graph
    # edges = set()
    # for [i, j, k] in mesh.faces:
    #     edges.add((i, j))
    #     edges.add((j, k))
    #     edges.add((k, i))
    # edge_index = []
    # for [i, j, k] in mesh.faces:
    #     if (j, i) not in edges:
    #         edge_index.append([j, i])
    #         edge_index.append([i, j])
    #     if (k, j) not in edges:
    #         edge_index.append([j, k])
    #         edge_index.append([k, j])
    #     if (i, k) not in edges:
    #         edge_index.append([k, i])
    #         edge_index.append([i, k])
    #
    # G = buildGraph(mesh, edge_index)

    load_txt_data(1)
