import numpy as np
import scipy.sparse as sp
import torch
from .reader import *
from .Dijkstra import Dijkstra
import os
from collections import defaultdict
from torch_geometric.data import InMemoryDataset, Data
import scipy as sp
import random
from numpy import linalg as LA



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

################################### K means part #####################################

def update_centers(mesh, center_pos, parent_list, child_list, belonging):
    delta_list = []
    for p in range(len(center_pos)):
        c_pos = center_pos[p]
        c_id_list = [i for i, x in enumerate(belonging) if x == parent_list[p]]
        count = belonging.count(parent_list[p]) + 1
        sum = c_pos
        for c in c_id_list:
            sum = np.add(sum, mesh.vertices[child_list[c], :])
        sum = sum / (1.0 * count)
        min_dis = 100000.0
        new_c = -1
        for c in child_list:
            dis = np.linalg.norm(np.subtract(sum, mesh.vertices[c, :]))
            if dis < min_dis:
                min_dis = dis
                new_c = c
        dis2 = np.linalg.norm(np.subtract(sum, c_pos))
        if min_dis < dis2:
            center_pos[p] = mesh.vertices[new_c, :]
            parent_list[p] = new_c
        delta = np.linalg.norm(np.subtract(center_pos[p], c_pos))
        delta_list.append(delta)
    return center_pos, np.linalg.norm(delta_list), parent_list


def get_mesh_map(mesh):
    map = np.zeros((mesh.num_vertices, mesh.num_vertices))
    mesh.enable_connectivity()
    for p in range(mesh.num_vertices):
        adj_v = mesh.get_vertex_adjacent_vertices(p)
        for j in range(adj_v.shape[0]):
            n1 = p
            n2 = adj_v[j]
            p1 = mesh.vertices[n1, :]
            p2 = mesh.vertices[n2, :]
            dp = LA.norm(p1 - p2)
            map[n1][n2] = map[n2][n1] = dp
    map_list = map.tolist()
    return map_list


def K_means(mesh, k):
    center_pos = []
    whole_list = [n for n in range(0, mesh.num_vertices)]
    parent_list = random.sample(range(0, mesh.num_vertices), k)
    child_list = [x for x in whole_list if x not in parent_list]
    belonging = [None] * len(child_list)  # length: child
    for p in parent_list:
        center_pos.append(mesh.vertices[p, :])
    norm_d = 10000.0
    map_list = get_mesh_map(mesh)
    Graph = Dijkstra(mesh.num_vertices, map_list, False)
    while norm_d > 1.0:
        t = 0
        for item1 in child_list:
            min_dis = 100000.0
            parent_id = -1
            for p in parent_list:
                dis = Graph.dijkstra2node(item1, p)
                if dis < min_dis:
                    parent_id = p
                    min_dis = dis
            belonging[t] = parent_id
            t = t + 1
        center_pos, norm_d, parent_list = update_centers(mesh, center_pos, parent_list, child_list, belonging)
        child_list = [x for x in whole_list if x not in parent_list]  # update child
    t = 0
    for item1 in child_list:
        min_dis = 100000.0
        parent_id = -1
        for p in parent_list:
            dis = Graph.dijkstra2node(item1, p)
            if dis < min_dis:
                parent_id = p
                min_dis = dis
        belonging[t] = parent_id
        t = t + 1

    return center_pos, child_list, parent_list, belonging






