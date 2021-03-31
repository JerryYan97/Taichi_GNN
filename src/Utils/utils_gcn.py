import numpy as np
import scipy.sparse as sp
import torch
import multiprocessing as mp
import pickle
import os
from collections import defaultdict
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import Dataset
import scipy as sp
from scipy.spatial import KDTree
from collections import Counter
from numpy import linalg as LA
from scipy.sparse import lil_matrix
from torch import linalg as torch_LA


# Each sample in it is in PyG format
def mp_load_global_data(workload_list, proc_idx,
                        filepath, files, node_num, edge_idx, culled_idx, cluster, transform, dim):
    sample_list = []
    for idx in range(workload_list[proc_idx][0], workload_list[proc_idx][1] + 1):
        fperframe = np.genfromtxt(filepath + "/" + files[idx], delimiter=',')
        # print("name: ", files[idx])
        if dim == 2:
            other = fperframe[:, 4:]
            pn_dis = fperframe[:, 2:4]
            pd_dis = fperframe[:, 0:2]  # a[start:stop] items start through stop-1
        else:
            # print("name: ", files[idx], " shape: ", fperframe.shape)
            # other = fperframe[:, 6:]
            pn_dis = fperframe[culled_idx, 3:6]
            pd_dis = fperframe[culled_idx, 0:3]  # a[start:stop] items start through stop-1
            other_tmp = fperframe[culled_idx, :]
            feat_idx = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23])
            other = np.take(other_tmp, feat_idx, axis=1)
        y_data = torch.from_numpy(np.subtract(pn_dis, pd_dis).reshape((node_num, -1)))
        x_data = torch.from_numpy(np.hstack((pd_dis, other)).reshape((node_num, -1)))
        sample = Data(x=x_data, edge_index=edge_idx, y=y_data, cluster=cluster)
        if transform:
            sample = transform(sample)
        sample_list.append(sample)

    print("proc", proc_idx, "-- start idx:", workload_list[proc_idx][0], " end idx:", workload_list[proc_idx][1])
    return sample_list


# Each sample in it is in normal PyTorch format.
# filepath is used to determine whether read files from training data folder or testing data folder.
# It won't append global feature vector to each sample, because it will blow up the RAM.
def mp_load_local_data(workload_list, mode, proc_idx, filepath, files, culled_bd_idx, culled_idx, dim):
    sample_list = []
    culled_bd_pts_num = culled_bd_idx.shape[0]
    culled_pts_num = len(culled_idx)
    gvec_dir = os.getcwd()
    if mode == 0:
        gvec_dir += "/SimData/TrainPreGenGlobalFeatureVec/"
    elif mode == 1:
        gvec_dir += "/SimData/TestPreGenGlobalFeatureVec/"

    for idx in range(workload_list[proc_idx][0], workload_list[proc_idx][1] + 1):
        fperframe = np.genfromtxt(filepath + "/" + files[idx], delimiter=',')
        if dim == 2:
            other = fperframe[culled_bd_idx, 4:]
            pn_dis = fperframe[culled_bd_idx, 2:4]
            pd_dis = fperframe[culled_bd_idx, 0:2]  # a[start:stop] items start through stop-1
        else:
            feat_idx = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23])

            # For the purpose of local training data:
            other_tmp = fperframe[culled_bd_idx, :]
            other = np.take(other_tmp, feat_idx, axis=1)
            pn_dis = fperframe[culled_bd_idx, 3:6]
            pd_dis = fperframe[culled_bd_idx, 0:3]  # a[start:stop] items start through stop-1

            # To produce global feature data:
            other_tmp_full = fperframe[culled_idx, :]
            other_full = np.take(other_tmp_full, feat_idx, axis=1)
            pn_dis_full = fperframe[culled_idx, 3:6]
            pd_dis_full = fperframe[culled_idx, 0:3]

        y_frame_data = torch.from_numpy(np.subtract(pn_dis, pd_dis).reshape((culled_bd_pts_num, -1)))
        x_frame_data = torch.from_numpy(np.hstack((pd_dis, other)).reshape((culled_bd_pts_num, -1)))

        y_frame_full_data = torch.from_numpy(np.subtract(pn_dis_full, pd_dis_full).reshape((culled_pts_num, -1)))
        x_frame_full_data = torch.from_numpy(np.hstack((pd_dis_full, other_full)).reshape((culled_pts_num, -1)))

        sample = {'x_frame': x_frame_data,
                  'y_frame': y_frame_data,
                  'x_frame_full': x_frame_full_data,
                  'y_frame_full': y_frame_full_data,
                  'filename': files[idx]}
        sample_list.append(sample)

    print("proc", proc_idx, "-- start idx:", workload_list[proc_idx][0], " end idx:", workload_list[proc_idx][1])
    return sample_list


class SIM_Data_Local(Dataset):
    def __init__(self, filepath, global_nn, mode, i_features_num, o_features_num, device, edge_idx, culled_idx,
                 culled_cluster, cluster_num,
                 global_culled_boundary_points_id, local_culled_boundary_points_id, dim,
                 include_global_vec=True, transform=None):
        # Read file names
        self._files = []
        for _, _, files in os.walk(filepath):
            self._files.extend(files)
        self._files.sort()
        # Set init parameters
        self._filepath = filepath
        self._input_features_num = i_features_num
        self._output_features_num = o_features_num
        self._local_culled_boundary_node_num = len(local_culled_boundary_points_id)
        self._cluster_num = cluster_num
        self._local_culled_boundary_pts_id = torch.from_numpy(local_culled_boundary_points_id)
        self._include_global_vec = include_global_vec
        # Read file data
        # Divide workloads:
        cpu_cnt = os.cpu_count()
        print("cpu core account: ", cpu_cnt)
        files_cnt = len(self._files)
        files_per_proc_cnt = files_cnt // cpu_cnt
        workload_list = []
        for i in range(cpu_cnt):
            # [[proc1 first file idx, proc1 last file idx] ... []]
            cur_proc_workload = [i * files_per_proc_cnt, (i + 1) * files_per_proc_cnt - 1]
            if i == cpu_cnt - 1:
                # Last workload may needs to do more than others.
                cur_proc_workload[1] = files_cnt - 1
            workload_list.append(cur_proc_workload)

        # Call multi-processing func to load samples:
        pool = mp.Pool()
        proc_list = []
        self._sample_list = []
        for i in range(cpu_cnt):
            proc_list.append(pool.apply_async(func=mp_load_local_data,
                                              args=(workload_list, mode, i, self._filepath, self._files,
                                                    local_culled_boundary_points_id, culled_idx, dim,)))

        # Get multi-processing res:
        for i in range(cpu_cnt):
            sub_sample_list = proc_list[i].get()
            self._sample_list.extend(sub_sample_list)
        pool.close()
        pool.join()

        print("Sample list length:", len(self._sample_list))

        # Calculate the global vec for each frame:
        # Put Data into NN:
        global_nn.eval()
        culled_node_num = len(culled_idx)
        batch = torch.zeros(culled_node_num)
        metric1 = 0.0
        small_cnt = 0
        if include_global_vec:
            print("Calculating the Global feature...")
            for i in range(len(self._sample_list)):
                with torch.no_grad():
                    g_feat, o_feat = global_nn(self._sample_list[i]['x_frame_full'].float().to(device),
                                               edge_idx.to(device),
                                               batch.to(device),
                                               culled_cluster.to(device))
                    y_data = self._sample_list[i]['y_frame_full']
                    output = o_feat.reshape(culled_node_num, -1)
                    output_cpu = output.cpu().detach()
                    top_vec = torch_LA.norm(output_cpu - y_data, dim=1).numpy()
                    bottom_vec = (torch_LA.norm(y_data, dim=1)).cpu().detach().numpy()
                    top_vec_b = np.take(top_vec, global_culled_boundary_points_id)
                    bottom_vec_b = np.take(bottom_vec, global_culled_boundary_points_id)
                    big_idx = np.where(bottom_vec_b > 1e-10)
                    top_cull_vec = np.take(top_vec_b, big_idx)
                    bottom_cull_vec = np.take(bottom_vec_b, big_idx)
                    tmp = top_cull_vec / bottom_cull_vec
                    if np.isinf(tmp).any():
                        raise Exception('Contain Elements that are inf!')
                    small_cnt += (len(top_vec_b) - len(big_idx[0]))
                    metric1 += np.sum(tmp)
                    # Save data to sample_list:
                    self._sample_list[i]["gvec"] = g_feat.cpu().detach()
            metric1 /= (len(global_culled_boundary_points_id) * len(self._files) - small_cnt)
            print("Avg metric1:", metric1)
        else:
            print("Doesn't include Global feature.")

    def __len__(self):
        return len(self._files) * self._local_culled_boundary_node_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_idx = idx // self._local_culled_boundary_node_num
        node_idx = idx - self._local_culled_boundary_node_num * file_idx
        try:
            x_frame_node = self._sample_list[file_idx]['x_frame'][node_idx]
        except:
            print("file_idx:", file_idx, " node_idx:", node_idx, " idx:", idx)
            exit(1)
        if self._include_global_vec:
            gvec_node = self._sample_list[file_idx]['gvec']
            x_data = torch.cat((x_frame_node, gvec_node), dim=0)
        else:
            x_data = x_frame_node
        # x_data = x_frame_node
        y_data = self._sample_list[file_idx]['y_frame'][node_idx]
        sample = {'x': x_data,
                  'y': y_data,
                  'filename': self._sample_list[file_idx]['filename'],
                  'mesh_vert_idx': self._local_culled_boundary_pts_id[node_idx],
                  'file_idx:': file_idx}
        return sample

    @property
    def input_features_num(self):
        return self._input_features_num

    @property
    def output_features_num(self):
        return self._output_features_num

    @property
    def boundary_node_num(self):
        return self._local_culled_boundary_node_num

    @property
    def boundary_node_mesh_idx(self):
        return self._local_culled_boundary_pts_id

    def len(self):
        return len(self._files)

    def to_device(self, device):
        import time
        to_d_start = time.time()
        for item in self._sample_list:
            item['x_frame'] = item['x_frame'].float().to(device)
            item['y_frame'] = item['y_frame'].float().to(device)
            item['gvec'] = item['gvec'].float().to(device)
        print("Dataset to device time:", time.time() - to_d_start)


class SIM_Data_Geo(InMemoryDataset):
    def __init__(self, filepath, mesh_edge_idx,
                 i_features_num, o_features_num,
                 case_info, cluster, clusters_num, dim,
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
        # self._node_num = case_info['mesh_num_vert']
        self._node_num = case_info['mesh_num_dyn_vert']

        ori_b_pt_idx = np.sort(np.fromiter(case_info['boundary'][0], int))
        hash_table = case_info['hash_table']
        new_b_pt_list = []
        for i in range(len(ori_b_pt_idx)):
            if hash_table.get(ori_b_pt_idx[i]) is not None:
                new_b_pt_list.append(hash_table[ori_b_pt_idx[i]])

        self._b_pt_idx = torch.from_numpy(np.array(new_b_pt_list))

        self._output_features_num = o_features_num

        self._cluster = cluster
        self._cluster_num = clusters_num

        # Section 5-3 multi-processing
        pool = mp.Pool()
        sample_list = []
        # Divide workloads:
        cpu_cnt = os.cpu_count()
        print("cpu core account: ", cpu_cnt)
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
            proc_list.append(pool.apply_async(func=mp_load_global_data,
                                              args=(workload_list, i, self._filepath, self._files, self.node_num,
                                                    self._edge_idx, case_info['culled_idx'], self._cluster,
                                                    self.transform, dim,)))
        # Get multi-processing res:
        for i in range(cpu_cnt):
            # print("i: ", i, ", get shape: ", len(proc_list[i].get()))
            sample_list.extend(proc_list[i].get())

        pool.close()
        pool.join()

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

    @property
    def boundary_pt_idx(self):
        return self._b_pt_idx

    def len(self):
        return len(self.raw_file_names)


# 0 -- train, 1 -- test
def load_local_data(case_info, hash_table, edge_idx, culled_idx, culled_cluster,
                    simulator_feature_num, global_feature_num, culled_cluster_num,
                    global_nn, device, mode=0, path="/Outputs", include_global_vec=True):
    file_dir = os.getcwd()
    file_dir = file_dir + path
    ori_boundary_pt_idx = case_info['boundary'][0]
    ori_boundary_pt_idx = np.sort(np.fromiter(ori_boundary_pt_idx, int))
    # NOTE: The culled boundary index in local is different from the culled boundary index in global.
    # The boundary index for local should be index for pts in simulator data which excludes fixed pts, because it is
    # used to read training data.
    #
    # The boundary index for global should be index for pts in culled fixed pts data, because it is used to calculate
    # the metric1 for output of the global NN.
    global_culled_boundary_pt_list = []
    local_culled_boundary_pt_list = []
    for i in range(len(ori_boundary_pt_idx)):
        if hash_table.get(ori_boundary_pt_idx[i]) is not None:
            global_culled_boundary_pt_list.append(hash_table[ori_boundary_pt_idx[i]])
            local_culled_boundary_pt_list.append(ori_boundary_pt_idx[i])
    global_culled_boundary_pt_idx = np.asarray(global_culled_boundary_pt_list)
    local_culled_boundary_pt_idx = np.asarray(local_culled_boundary_pt_list)

    total_feature_num = simulator_feature_num
    if include_global_vec:
        total_feature_num += global_feature_num

    tmp_dataset = SIM_Data_Local(file_dir, global_nn, mode, total_feature_num, 3, device, edge_idx, culled_idx,
                                 culled_cluster, culled_cluster_num,
                                 global_culled_boundary_pt_idx,
                                 local_culled_boundary_pt_idx, case_info['dim'], include_global_vec)
    return tmp_dataset


# file_dir: Top folder path.
def load_cluster(file_dir, test_case, cluster_num, vert_num):
    filename = file_dir+"/MeshModels/SavedClusters/"+"test_case"+str(test_case)+"_c"+str(cluster_num)+"_cluster"+".csv"
    cluster = np.genfromtxt(filename, delimiter=',', dtype=np.float)
    cluster_num = int(cluster[0])
    belonging = torch.tensor(cluster[1:1+vert_num].astype(int))
    belonging_len = cluster[1+vert_num:1+2*vert_num]
    parent_np = cluster[1+2*vert_num:1+2*vert_num+cluster_num].astype(int)
    cluster_parent = np.zeros(vert_num, dtype=np.int)
    cluster_belongs = []
    for a in range(cluster_num):
        cluster_belongs.append([])
    for i in range(vert_num):
        belong_idx = belonging[i]
        cluster_parent[i] = parent_np[belong_idx]
        cluster_belongs[belong_idx].append(i)
    return belonging, cluster_num, cluster_parent, cluster_belongs, belonging_len


def load_global_info(case_info, case_id, cluster_num):
    full_idx = np.arange(case_info['mesh_num_vert'])
    culled_idx = np.delete(full_idx, case_info['dirichlet'])
    hash_table = {}
    for i in range(len(culled_idx)):
        hash_table[culled_idx[i]] = i
    cluster, _, _, _, _ = load_cluster(os.getcwd(), case_id, cluster_num, case_info['mesh_num_vert'])
    cluster = cluster[culled_idx]
    unique_num = len(np.unique(cluster))
    edges = set()
    edge_index = np.zeros(shape=(2, 0), dtype=np.int32)
    for [i, j, k, m] in case_info['elements']:
        if i not in case_info['dirichlet'] and j not in case_info['dirichlet']:
            edges.add((hash_table[i], hash_table[j]))
            edges.add((hash_table[j], hash_table[i]))
        if i not in case_info['dirichlet'] and k not in case_info['dirichlet']:
            edges.add((hash_table[i], hash_table[k]))
            edges.add((hash_table[k], hash_table[i]))
        if i not in case_info['dirichlet'] and m not in case_info['dirichlet']:
            edges.add((hash_table[i], hash_table[m]))
            edges.add((hash_table[m], hash_table[i]))
        if j not in case_info['dirichlet'] and k not in case_info['dirichlet']:
            edges.add((hash_table[j], hash_table[k]))
            edges.add((hash_table[k], hash_table[j]))
        if j not in case_info['dirichlet'] and m not in case_info['dirichlet']:
            edges.add((hash_table[j], hash_table[m]))
            edges.add((hash_table[m], hash_table[j]))
        if k not in case_info['dirichlet'] and m not in case_info['dirichlet']:
            edges.add((hash_table[k], hash_table[m]))
            edges.add((hash_table[m], hash_table[k]))
    for (i, j) in edges:
        edge_index = np.hstack((edge_index, [[i], [j]]))
    edge_index = torch.LongTensor(edge_index)
    return unique_num, len(culled_idx), edge_index, hash_table, cluster, culled_idx


# Load data record:
# case 1001 -- 9.8G (Without optimization):
# t1: 0.003854036331176758  t2: 0.03281879425048828  t3: 0.00013327598571777344  t4: 0.0012357234954833984
def load_global_data(test_case, cluster_num, path="/Outputs"):
    file_dir = os.getcwd()
    file_dir = file_dir + path
    print(os.getcwd() + "/MeshModels/MeshInfo/case_info" + str(test_case))
    case_info = pickle.load(open(os.getcwd() + "/MeshModels/MeshInfo/case_info" + str(test_case) + ".p", "rb"))
    # case_info = read(test_case)
    # mesh = case_info['mesh']

    edges = set()
    edge_index = np.zeros(shape=(2, 0), dtype=np.int32)
    if case_info['dim'] == 2:
        raise Exception("Doesn't support 2D now.")
        # for [i, j, k] in mesh.faces:
        #     edges.add((i, j))
        #     edges.add((j, k))
        #     edges.add((k, i))
        # for [i, j, k] in mesh.faces:
        #     if (j, i) not in edges:
        #         edge_index = np.hstack((edge_index, [[j], [i]]))
        #         edge_index = np.hstack((edge_index, [[i], [j]]))
        #     if (k, j) not in edges:
        #         edge_index = np.hstack((edge_index, [[j], [k]]))
        #         edge_index = np.hstack((edge_index, [[k], [j]]))
        #     if (i, k) not in edges:
        #         edge_index = np.hstack((edge_index, [[k], [i]]))
        #         edge_index = np.hstack((edge_index, [[i], [k]]))
        # edge_index = torch.LongTensor(edge_index)
        # cluster, cluster_num, cluster_parent, belongs, belongs_len = load_cluster(os.getcwd(), test_case, cluster_num, mesh.num_vertices)
        # # Note: boundary labels are not feed into the dataset.
        # return SIM_Data_Geo(file_dir, edge_index, 14, 2, mesh, cluster, cluster_num, cluster_parent, belongs, 2)
    else:
        import time
        t1_start = time.time()

        # Construct full idx to fixed index cull hashtable:
        hash_table = {}
        full_idx = np.arange(case_info['mesh_num_vert'])
        culled_idx = np.delete(full_idx, case_info['dirichlet'])
        for i in range(len(culled_idx)):
            hash_table[culled_idx[i]] = i
        case_info['culled_idx'] = culled_idx
        case_info['mesh_num_dyn_vert'] = len(culled_idx)
        case_info['hash_table'] = hash_table

        # Load Section 1
        for [i, j, k, m] in case_info['elements']:
            if i not in case_info['dirichlet'] and j not in case_info['dirichlet']:
                edges.add((hash_table[i], hash_table[j]))
                edges.add((hash_table[j], hash_table[i]))
            if i not in case_info['dirichlet'] and k not in case_info['dirichlet']:
                edges.add((hash_table[i], hash_table[k]))
                edges.add((hash_table[k], hash_table[i]))
            if i not in case_info['dirichlet'] and m not in case_info['dirichlet']:
                edges.add((hash_table[i], hash_table[m]))
                edges.add((hash_table[m], hash_table[i]))
            if j not in case_info['dirichlet'] and k not in case_info['dirichlet']:
                edges.add((hash_table[j], hash_table[k]))
                edges.add((hash_table[k], hash_table[j]))
            if j not in case_info['dirichlet'] and m not in case_info['dirichlet']:
                edges.add((hash_table[j], hash_table[m]))
                edges.add((hash_table[m], hash_table[j]))
            if k not in case_info['dirichlet'] and m not in case_info['dirichlet']:
                edges.add((hash_table[k], hash_table[m]))
                edges.add((hash_table[m], hash_table[k]))

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
        cluster, cluster_num, cluster_parent, belongs, belongs_len = load_cluster(os.getcwd(), test_case, cluster_num,
                                                                                  case_info['mesh_num_vert'])
        t4_end = time.time()
        print("t4:", t4_end - t4_start)

        # Load Section 5
        t5_start = time.time()
        cluster = cluster[culled_idx]
        unique_num = len(np.unique(cluster))
        if unique_num != cluster_num:
            print("Warning: Culled Cluster Number is smaller than the original Cluster Number.")
            print("The original cluster number is:", cluster_num)
            print("The culled cluster number is:", unique_num)
        tmp_data = SIM_Data_Geo(file_dir, edge_index, 18, 3,
                                case_info,
                                cluster, unique_num, 3)
        t5_end = time.time()
        print("t5:", t5_end - t5_start)

        return tmp_data, case_info, cluster_parent, belongs


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


def spt_parallel_func2(workload_list, proc_idx, shared_adj_mat, src_list, vertices_num):
    dist_list = []
    for idx in range(workload_list[proc_idx][0], workload_list[proc_idx][1]):
        dist = [float("inf")] * vertices_num
        dist[src_list[idx]] = 0
        min_dist_set = [False] * vertices_num
        for count in range(vertices_num):
            # if proc_idx == 0:
            #     if count % 10 == 0:
            #         print("finished:", float(count) * 100.0 / float(vertices_num))
            # minimum distance vertex that is not processed
            u = min_distance(vertices_num, dist, min_dist_set)
            # put minimum distance vertex in shortest tree
            min_dist_set[u] = True
            # Update dist value of the adjacent vertices
            for v in range(vertices_num):
                if shared_adj_mat[u, v] > 0 and min_dist_set[v] == False and dist[v] > dist[u] + shared_adj_mat[u, v]:
                    dist[v] = dist[u] + shared_adj_mat[u, v]
        dist_list.append(dist)
    return dist_list


def belonging_parallel_func(workload_list, proc_idx, spt_list, cluster_num):
    belonging_list = []
    belonging_len_list = []
    for vert_idx in range(workload_list[proc_idx][0], workload_list[proc_idx][1] + 1):
        min_dis = 100000.0
        cluster_idx = -1
        for i in range(cluster_num):
            dis = spt_list[i][vert_idx]
            if dis < min_dis:
                cluster_idx = i
                min_dis = dis
        belonging_list.append(cluster_idx)
        belonging_len_list.append(min_dis)
    return belonging_list, belonging_len_list


class MeshKmeansHelper2():
    def __init__(self, cluster_num, vertices_num, adj_mat, v_pos):
        self._kdTree = KDTree(v_pos)
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
        # Divide workloads
        cpu_cnt = os.cpu_count()
        total_works = self.k
        rest_works = self.k
        min_works_per_proc_cnt = total_works // cpu_cnt
        max_works_per_proc_cnt = min_works_per_proc_cnt + 1
        workloads_list = []
        proc_list = []
        for i in range(cpu_cnt):
            rest_cpu = cpu_cnt - i  # Include itself.
            first_need_assigned_work = total_works - rest_works
            if rest_works == min_works_per_proc_cnt * rest_cpu:
                # Do min works
                cur_proc_workload = [first_need_assigned_work, first_need_assigned_work + min_works_per_proc_cnt]
                rest_works -= min_works_per_proc_cnt
            else:
                # Do max works
                cur_proc_workload = [first_need_assigned_work, first_need_assigned_work + max_works_per_proc_cnt]
                rest_works -= max_works_per_proc_cnt
            workloads_list.append(cur_proc_workload)

        # Parallel call
        for t in range(cpu_cnt):
            proc_list.append(pool.apply_async(func=spt_parallel_func2,
                                              args=(workloads_list, t, self._adj_mat,
                                                    self._src_list, self._vertices_num,)))
        # Get results
        for t in range(cpu_cnt):
            self._spt_list.extend(proc_list[t].get())

    def generate_belongs(self, pool):
        belonging = []
        belonging_len = []
        # Divide workloads
        cpu_cnt = os.cpu_count()
        works_per_proc_cnt = self._vertices_num // cpu_cnt
        workloads_list = []
        proc_list = []
        for i in range(cpu_cnt):
            cur_proc_workload = [i * works_per_proc_cnt, (i + 1) * works_per_proc_cnt - 1]
            if i == cpu_cnt - 1:
                cur_proc_workload[1] = self._vertices_num - 1
            workloads_list.append(cur_proc_workload)
        # Parallel call
        for t in range(cpu_cnt):
            proc_list.append(pool.apply_async(func=belonging_parallel_func,
                                              args=(workloads_list, t, self._spt_list, self._k)))
        # Get results
        for t in range(cpu_cnt):
            local_belonging, local_belonging_len = proc_list[t].get()
            belonging.extend(local_belonging)
            belonging_len.extend(local_belonging_len)
        return belonging, belonging_len

    def update_center(self, parent_list, belong_list, v_pos):
        center_pos = np.zeros(shape=(self._k, 3), dtype=np.float)
        cluster_cnt = np.zeros(self._k, dtype=np.int)
        # Calculate center
        for i in range(self._vertices_num):
            cluster_belong = belong_list[i]
            center_pos[cluster_belong] += v_pos[i]
            cluster_cnt[cluster_belong] += 1
        for i in range(self._k):
            center_pos[i] /= cluster_cnt[i]
        # Find new center
        dd, ii = self._kdTree.query(center_pos)

        # Check duplicate cluster center:
        # https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
        tmp = [item for item, count in Counter(ii).items() if count > 1]
        if len(tmp) != 0:
            raise Exception("There is a duplicate cluster center.")

        parent_list_new = np.asarray(ii, dtype=np.int)
        change_amount = 0.0
        for i in range(self._k):
            center_old = v_pos[parent_list[i]]
            center_new = v_pos[parent_list_new[i]]
            change_amount += LA.norm(center_old - center_new)
        return parent_list_new, change_amount

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


# New algorithm and add in length to their parents
# Algorithm: https://yfzhong.wordpress.com/2014/11/25/using-k-means-algorithm-for-mesh-and-image-segmentation-matlab/
# NOTE: This is deprecated. The most expensive part of it is the spt tree construction.
# It doesn't have a good method to construct 128/256/512 spt tree in cpu in a short time. So, I will just try whether
# Taichi can give better performance. Besides, cuGraph maybe also a good place to try.
def K_means_multiprocess2(mesh, k):
    if k > mesh.num_vertices:
        raise Exception("k should be less than mesh's vertices num.")

    if k < os.cpu_count():
        raise Exception("Currently it doesn't support clusters num less than cpu cores num.")

    # Two adjustable parameters to control the convergence:
    max_itr = 10
    convergence_rate = 0.9
    bbox_diag_len = LA.norm(mesh.bbox[0] - mesh.bbox[1])

    whole_list = [n for n in range(0, mesh.num_vertices)]
    parent_list = [i for i in range(0, mesh.num_vertices, (mesh.num_vertices // k) + 1)]
    belonging = np.arange(mesh.num_vertices, dtype=np.int)
    pool = mp.Pool()

    kmeans_helper = MeshKmeansHelper2(k, mesh.num_vertices, get_mesh_map(mesh), mesh.vertices)

    for itr in range(max_itr):
        # assignment each point to its nearest center
        kmeans_helper.generate_spt_list(parent_list, pool)
        belonging, belonging_len = kmeans_helper.generate_belongs(pool)
        # recalculate the center for each cluster
        parent_list_last = parent_list
        parent_list, change_amount = kmeans_helper.update_center(parent_list, belonging, mesh.vertices)
        print("Total center pos change amount (Less is better):", change_amount)
        if (1.0 - convergence_rate) * bbox_diag_len > change_amount:
            break

    pool.close()
    pool.join()

    return parent_list_last, belonging, belonging_len
