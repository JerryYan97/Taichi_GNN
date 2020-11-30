import numpy as np
import scipy.sparse as sp
import torch
from .reader import *
import os
# use data loader
from torch.utils.data import Dataset
import matplotlib
from collections import defaultdict


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
                
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print(adj.todense())

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # sp.eye 对角矩阵 adj+I

    idx_train = range(140)
    idx_val = range(140, 350)
    idx_test = range(350, 500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


class SIM_Data(Dataset):
    def __init__(self, filepath, transform=None):
        Files_Global = []
        for _, _, files in os.walk(filepath):
            Files_Global.append(files)
        Files_Global = Files_Global[0]
        Files_Global.sort()
        inputs = np.zeros((0, 0))
        outputs = np.zeros((0, 0))

        for f in range(len(Files_Global)):  # the data in each frame
            fperframe = np.genfromtxt("{}{}".format(filepath + "/", Files_Global[f]), dtype=np.dtype(str))
            other = fperframe[:, 4:].astype(float)
            pn_dis = fperframe[:, 2:4].astype(float)
            pd_dis = fperframe[:, 0:2].astype(float)  # a[start:stop] items start through stop-1
            pos_delta = np.subtract(pn_dis, pd_dis).reshape((1, -1))
            pn_dis = pn_dis.reshape((1, -1))
            withoutpn = np.hstack((pd_dis, other)).reshape((1, -1))
            if f == 0:
                inputs = withoutpn  # .reshape((1, -1))
                # outputs = pos_delta  # .reshape((1, -1))
                outputs = pn_dis
            else:
                inputs = np.vstack((inputs, withoutpn))
                # outputs = np.vstack((outputs, pos_delta))
                outputs = np.vstack((outputs, pn_dis))

        self.n_samples = inputs.shape[0]
        self.x_data = inputs
        self.y_data = outputs
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, outputs = sample
        return torch.from_numpy(inputs), torch.from_numpy(outputs)


def load_txt_data2(objpath, path="Outputs"):
    """Load citation network dataset (cora only for now)"""
    file_dir = os.getcwd()
    file_dir = file_dir + "/Outputs"
    print('\nWith Tensor Transform')
    dataset = SIM_Data(file_dir, transform=ToTensor())
    first_data = dataset[0]
    features, labels = first_data
    print(features.shape, "          ",  labels.shape)

    mesh, _, _, _ = read(int(objpath))  # build graph
    edges = set()
    for [i, j, k] in mesh.faces:
        edges.add((i, j))
        edges.add((j, k))
        edges.add((k, i))
    boundary_points_ = set()
    boundary_edges_ = np.zeros(shape=(0, 2), dtype=np.int32)
    edge_index = np.zeros(shape=(2, 0), dtype=np.int32)
    for [i, j, k] in mesh.faces:
        if (j, i) not in edges:
            boundary_points_.update([j, i])
            boundary_edges_ = np.vstack((boundary_edges_, [j, i]))
            edge_index = np.hstack((edge_index, [[j], [i]]))
            edge_index = np.hstack((edge_index, [[i], [j]]))
        if (k, j) not in edges:
            boundary_points_.update([k, j])
            boundary_edges_ = np.vstack((boundary_edges_, [k, j]))
            edge_index = np.hstack((edge_index, [[j], [k]]))
            edge_index = np.hstack((edge_index, [[k], [j]]))
        if (i, k) not in edges:
            boundary_points_.update([i, k])
            boundary_edges_ = np.vstack((boundary_edges_, [i, k]))
            edge_index = np.hstack((edge_index, [[k], [i]]))
            edge_index = np.hstack((edge_index, [[i], [k]]))

    v_num = mesh.num_vertices
    row, col, data = [], [], []
    mesh.enable_connectivity()  # build graph
    for i in range(v_num):
        adjv = mesh.get_vertex_adjacent_vertices(i)
        for j in range(adjv.shape[0]):
            data = np.append(data, 1)
            row = np.append(row, i)
            col = np.append(col, adjv[j])
    # print("row: ", row)
    # print("col: ", col)
    # print("data: ", data)
    # coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    adj = sp.coo_matrix((data, (row, col)), shape=(v_num, v_num), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # build symmetric adjacency matrix

    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    edge_index = torch.LongTensor(edge_index)

    return mesh, adj, edge_index, dataset


def load_st_txt_data(objpath, path="Outputs_T"):
    """Load citation network dataset (cora only for now)"""
    file_dir = os.getcwd()
    file_dir = file_dir + "/Outputs_T"
    print('\nWith Tensor Transform')
    dataset = SIM_Data(file_dir, transform=ToTensor())

    first_data = dataset[0]
    features, labels = first_data
    # print(type(features), type(labels))
    print(features.shape, "          ",  labels.shape)
    # print(features, labels)

    mesh, _, _, _ = read(int(objpath))  # build graph
    edges = set()
    for [i, j, k] in mesh.faces:
        edges.add((i, j))
        edges.add((j, k))
        edges.add((k, i))
    boundary_points_ = set()
    boundary_edges_ = np.zeros(shape=(0, 2), dtype=np.int32)
    edge_index = np.zeros(shape=(2, 0), dtype=np.int32)
    for [i, j, k] in mesh.faces:
        if (j, i) not in edges:
            boundary_points_.update([j, i])
            boundary_edges_ = np.vstack((boundary_edges_, [j, i]))
            edge_index = np.hstack((edge_index, [[j], [i]]))
            edge_index = np.hstack((edge_index, [[i], [j]]))
        if (k, j) not in edges:
            boundary_points_.update([k, j])
            boundary_edges_ = np.vstack((boundary_edges_, [k, j]))
            edge_index = np.hstack((edge_index, [[j], [k]]))
            edge_index = np.hstack((edge_index, [[k], [j]]))
        if (i, k) not in edges:
            boundary_points_.update([i, k])
            boundary_edges_ = np.vstack((boundary_edges_, [i, k]))
            edge_index = np.hstack((edge_index, [[k], [i]]))
            edge_index = np.hstack((edge_index, [[i], [k]]))

    edge_index = torch.LongTensor(edge_index)

    return mesh, edge_index, dataset


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

    load_txt_data2(1)
    # load_test_txt_data()
