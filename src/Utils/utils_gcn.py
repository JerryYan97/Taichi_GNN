import numpy as np
import scipy.sparse as sp
import torch
import multiprocessing as mp
from .reader import *
from .Dijkstra import Dijkstra
import os
from collections import defaultdict
from torch_geometric.data import InMemoryDataset, Data
import scipy as sp
import random
from numpy import linalg as LA
from os import system
from scipy.sparse import lil_matrix


def mp_load_data(workload_list, proc_idx, filepath, files, node_num, edge_idx, cluster, transform, dim):
    sample_list = []
    for idx in range(workload_list[proc_idx][0], workload_list[proc_idx][1] + 1):
        fperframe = np.genfromtxt(filepath + "/" + files[idx], delimiter=',')
        if dim == 2:
            other = fperframe[:, 4:]
            pn_dis = fperframe[:, 2:4]
            pd_dis = fperframe[:, 0:2]  # a[start:stop] items start through stop-1
        else:
            other = fperframe[:, 6:]
            pn_dis = fperframe[:, 3:6]
            pd_dis = fperframe[:, 0:3]  # a[start:stop] items start through stop-1
        y_data = torch.from_numpy(np.subtract(pn_dis, pd_dis).reshape((node_num, -1)))
        x_data = torch.from_numpy(np.hstack((pd_dis, other)).reshape((node_num, -1)))
        sample = Data(x=x_data, edge_index=edge_idx, y=y_data, cluster=cluster)
        if transform:
            sample = transform(sample)
        sample_list.append(sample)

    print("proc", proc_idx, "-- start idx:", workload_list[proc_idx][0], " end idx:", workload_list[proc_idx][1])
    return sample_list


class SIM_Data_Geo(InMemoryDataset):
    def __init__(self, filepath, mesh_edge_idx,
                 i_features_num, o_features_num,
                 mesh, cluster, clusters_num, dim,
                 transform=None, pre_transform=None):
        super(SIM_Data_Geo, self).__init__(None, transform, pre_transform)
        import time

        # Section 5-1
        t5_1_start = time.time()
        self._files = []
        for _, _, files in os.walk(filepath):
            self._files.extend(files)
        t5_1_end = time.time()
        print("t5-1:", t5_1_end - t5_1_start)

        # Section 5-2
        t5_2_start = time.time()
        self._files.sort()
        t5_2_end = time.time()
        print("t5-2:", t5_2_end - t5_2_start)
        self._edge_idx = mesh_edge_idx
        self._filepath = filepath
        self._input_features_num = i_features_num
        self._node_num = mesh.num_vertices
        self._output_features_num = o_features_num

        self._cluster = cluster
        self._cluster_num = clusters_num

        # Section 5-3 multi-processing
        pool = mp.Pool()
        sample_list = []
        # Divide workloads:
        cpu_cnt = os.cpu_count()
        files_cnt = self.len()
        files_per_proc_cnt = files_cnt // cpu_cnt
        workload_list = []
        proc_list = []
        for i in range(cpu_cnt):
            # [[proc1 first file idx, proc1 last file idx] ... []]
            cur_proc_workload = [i * files_per_proc_cnt, (i + 1) * files_per_proc_cnt - 1]
            if i == cpu_cnt - 1:
                # Last workload may needs to do more than others.
                cur_proc_workload[1] = files_cnt - 1
            workload_list.append(cur_proc_workload)
        # Call multi-processing func:
        for i in range(cpu_cnt):
            proc_list.append(pool.apply_async(func=mp_load_data,
                                              args=(workload_list, i, self._filepath, self._files, self.node_num,
                                                    self._edge_idx, self._cluster, self.transform, dim,)))
        # Get multi-processing res:
        for i in range(cpu_cnt):
            sample_list.extend(proc_list[i].get())

        print("Sample list length:", len(sample_list))
        self.data, self.slices = self.collate(sample_list)

    @property
    def cluster(self):
        return self._cluster

    @property
    def cluster_num(self):
        return self._cluster_num

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


# file_dir: Top folder path.
def load_cluster(file_dir, test_case):
    cluster = np.genfromtxt(file_dir + "/MeshModels/SavedClusters/" + f"test_case{test_case}_cluster.csv",
                            delimiter=',', dtype=int)
    cluster_num = cluster[len(cluster) - 1]
    cluster = torch.tensor(cluster[:len(cluster) - 1])
    return cluster, cluster_num


# Load data record:
# case 1001 -- 9.8G (Without optimization):
# t1: 0.003854036331176758  t2: 0.03281879425048828  t3: 0.00013327598571777344  t4: 0.0012357234954833984
#
def load_data(test_case, path="/Outputs"):
    file_dir = os.getcwd()
    file_dir = file_dir + path

    case_info = read(test_case)
    mesh = case_info['mesh']

    edges = set()
    edge_index = np.zeros(shape=(2, 0), dtype=np.int32)
    if case_info['dim'] == 2:
        for [i, j, k] in mesh.faces:
            edges.add((i, j))
            edges.add((j, k))
            edges.add((k, i))
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
        cluster, cluster_num = load_cluster(os.getcwd(), test_case)
        return SIM_Data_Geo(file_dir, edge_index, 14, 2, mesh, cluster, cluster_num, 2)
    else:
        import time
        t1_start = time.time()
        # Load Section 1
        for [i, j, k, m] in mesh.elements:
            edges.add((i, j))
            edges.add((j, i))
            edges.add((i, k))
            edges.add((k, i))
            edges.add((i, m))
            edges.add((m, i))

            edges.add((j, k))
            edges.add((k, j))
            edges.add((j, m))
            edges.add((m, j))

            edges.add((k, m))
            edges.add((m, k))
        t1_end = time.time()

        t2_start = time.time()
        # Load Section 2
        for (i, j) in edges:
            edge_index = np.hstack((edge_index, [[i], [j]]))
        t2_end = time.time()

        print("t1:", t1_end - t1_start, " t2:", t2_end - t2_start)

        # Load Section 3
        t3_start = time.time()
        edge_index = torch.LongTensor(edge_index)
        t3_end = time.time()
        print("t3:", t3_end - t3_start)

        # Load Section 4
        t4_start = time.time()
        cluster, cluster_num = load_cluster(os.getcwd(), test_case)
        t4_end = time.time()
        print("t4:", t4_end - t4_start)

        # Load Section 5
        t5_start = time.time()
        tmp_data = SIM_Data_Geo(file_dir, edge_index, 24, 3, mesh, cluster, cluster_num, 3)
        t5_end = time.time()
        print("t5:", t5_end - t5_start)

        return tmp_data, case_info


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
    edges = []  # for each node in graph
    for node in graph:
        for neighbour in graph[node]:  # for each neighbour node of a single node
            edges.append((node, neighbour))  # if edge exists then append
    return edges


def addEdge(graph, u, v):
    graph[u].append(v)


# definition of function
def buildGraph(mesh, edge):
    graph = defaultdict(list)  # function for adding edge to graph
    for e in edge:  # declaration of graph as dictionary
        addEdge(graph, e[0], e[1])
    print(generate_edges(graph))  # Driver Function call， to print generated graph
    return graph


################################### K means part #####################################
def min_distance(vertices_num, dist, min_dist_set):
    min_dist = float("inf")
    for v in range(vertices_num):
        if dist[v] < min_dist and min_dist_set[v] == False:
            min_dist = dist[v]
            min_index = v
    return min_index


def spt_parallel_func(shared_adj_mat, src_list, vertices_num, idx):
    dist = [float("inf")] * vertices_num
    dist[src_list[idx]] = 0
    min_dist_set = [False] * vertices_num
    for count in range(vertices_num):
        # minimum distance vertex that is not processed
        u = min_distance(vertices_num, dist, min_dist_set)
        # put minimum distance vertex in shortest tree
        min_dist_set[u] = True
        # Update dist value of the adjacent vertices
        for v in range(vertices_num):
            if shared_adj_mat[u, v] > 0 and min_dist_set[v] == False and dist[v] > dist[u] + shared_adj_mat[u, v]:
                dist[v] = dist[u] + shared_adj_mat[u, v]
    return dist

# Old one
# def childlist_helper_parallel_func(childlist_item, parent_list, src_list, spt_list):
#     min_dis = 100000.0
#     parent_id = -1
#     for p in parent_list:
#         list_idx = src_list.index(p)
#         dis = spt_list[list_idx][childlist_item]
#         if dis < min_dis:
#             parent_id = p
#             min_dis = dis
#     return parent_id


def childlist_helper_parallel_func(workload_list, proc_idx, childlist, parent_list, src_list, spt_list):
    parent_id_list = []
    for childlist_idx in range(workload_list[proc_idx][0], workload_list[proc_idx][1] + 1):
        childlist_item = childlist[childlist_idx]
        min_dis = 100000.0
        parent_id = -1
        for p in parent_list:
            list_idx = src_list.index(p)
            dis = spt_list[list_idx][childlist_item]
            if dis < min_dis:
                parent_id = p
                min_dis = dis
        parent_id_list.append(parent_id)
    return parent_id_list


class MeshKmeansHelper():
    def __init__(self, cluster_num, vertices_num, adj_mat):
        self._k = cluster_num
        self._vertices_num = vertices_num
        self._adj_mat = adj_mat
        self._spt_list = []
        self._src_list = []

    def generate_spt_list(self, src_list, pool):
        if len(src_list) != self.k:
            raise Exception("Input srcs nums(", len(src_list), ") is not equal to k(", self.k, ").")
        # Init relevant lists
        self._spt_list = []
        self._src_list = src_list
        # Generate spt list
        res_list = []
        for i in range(self.k):
            res_list.append(pool.apply_async(func=spt_parallel_func,
                                             args=(self._adj_mat, self._src_list, self._vertices_num, i,)))
        for i in range(self.k):
            self._spt_list.append(res_list[i].get())

    def generate_belongs(self, child_list, parent_list, pool):
        belonging = []

        # Divide workloads
        cpu_cnt = os.cpu_count()
        child_list_len = len(child_list)
        works_per_proc_cnt = child_list_len // cpu_cnt
        workloads_list = []
        proc_list = []
        for i in range(cpu_cnt):
            cur_proc_workload = [i * works_per_proc_cnt, (i + 1) * works_per_proc_cnt - 1]
            if i == cpu_cnt - 1:
                cur_proc_workload[1] = child_list_len - 1
            workloads_list.append(cur_proc_workload)

        # Parallel call
        for t in range(cpu_cnt):
            proc_list.append(pool.apply_async(func=childlist_helper_parallel_func,
                                              args=(workloads_list, t, child_list, parent_list,
                                                    self._src_list, self._spt_list,)))
        # Get results
        for t in range(cpu_cnt):
            belonging.extend(proc_list[t].get())
        # Old
        # # Parallel call
        # for t in range(len(child_list)):
        #     res_list.append(pool.apply_async(func=childlist_helper_parallel_func,
        #                                      args=(child_list[t], parent_list, self._src_list, self._spt_list,)))
        # # Get results
        # for t in range(len(child_list)):
        #     belonging[t] = res_list[t].get()
        #     if t % 10 == 0:
        #         print("section 1 progress:", (float(t) / len(child_list)) * 100.0, "%")
        return belonging

    def get_dist(self, src_idx, dst_idx):
        list_idx = self._src_list.index(src_idx)
        return self._spt_list[list_idx][dst_idx]

    def get_spt(self, src_idx):
        list_idx = self._src_list.index(src_idx)
        return self._spt_list[list_idx]

    @property
    def k(self):
        return self._k

    @property
    def vertices_num(self):
        return self._vertices_num

    @property
    def adj_mat(self):
        return self._adj_mat


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
    map = lil_matrix((mesh.num_vertices, mesh.num_vertices), dtype=float)
    mesh.enable_connectivity()
    for p in range(mesh.num_vertices):
        adj_v = mesh.get_vertex_adjacent_vertices(p)
        for j in range(adj_v.shape[0]):
            n1 = p
            n2 = adj_v[j]
            p1 = mesh.vertices[n1]
            p2 = mesh.vertices[n2]
            dp = LA.norm(p1 - p2)
            map[n1, n2] = map[n2, n1] = dp
    return map


def K_means_multiprocess(mesh, k):
    center_pos = []
    whole_list = [n for n in range(0, mesh.num_vertices)]
    parent_list = [i for i in range(0, mesh.num_vertices, (mesh.num_vertices // k) + 1)]
    child_list = [x for x in whole_list if x not in parent_list]
    for p in parent_list:
        center_pos.append(mesh.vertices[p, :])
    norm_d = 10000.0
    pool = mp.Pool()

    cluster_helper = MeshKmeansHelper(k, mesh.num_vertices, get_mesh_map(mesh))

    while norm_d > 1.0:
        cluster_helper.generate_spt_list(parent_list, pool)
        belonging = cluster_helper.generate_belongs(child_list, parent_list, pool)
        center_pos, norm_d, parent_list = update_centers(mesh, center_pos, parent_list, child_list, belonging)
        child_list = [x for x in whole_list if x not in parent_list]  # update child

    cluster_helper.generate_spt_list(parent_list, pool)
    belonging = cluster_helper.generate_belongs(child_list, parent_list, pool)
    pool.close()
    pool.join()

    return center_pos, child_list, parent_list, belonging
