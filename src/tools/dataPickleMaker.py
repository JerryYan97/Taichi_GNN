import pickle
import sys
import os
import torch
import numpy as np
import time
import multiprocessing as mp
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.utils_gcn import load_cluster


def mp_load_data(start_file_idx, end_file_idx, proc_idx, edge_idx, filepath, files, culled_bd_idx, culled_idx):
    local_sample_list = []
    global_sample_list = []
    culled_bd_pts_num = culled_bd_idx.shape[0]
    culled_pts_num = len(culled_idx)

    print("proc", proc_idx, "-- start idx:", start_file_idx, " end idx:", end_file_idx)
    for idx in range(len(files)):
        fperframe = np.genfromtxt(filepath + "/" + files[idx], delimiter=',')
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

        local_sample = {'x_frame': x_frame_data,
                        'y_frame': y_frame_data,
                        'x_frame_full': x_frame_full_data,
                        'y_frame_full': y_frame_full_data,
                        'filename': files[idx]}
        global_sample = Data(x=x_frame_full_data, edge_index=edge_idx, y=y_frame_full_data, cluster=cluster)
        local_sample_list.append(local_sample)
        global_sample_list.append(global_sample)

    sample_list = {
        'local_sample_list': local_sample_list,
        'global_sample_list': global_sample_list
    }

    print("proc", proc_idx, " finish works.")
    print("First file:", files[0], ". Last file:", files[len(files) - 1])

    return sample_list


if __name__ == '__main__':
    load_data_t_start = time.time()
    # Read file names
    case_id = 1011
    cluster_num = 128
    additional_note = '7sets_data'
    case_info = pickle.load(open("../../MeshModels/MeshInfo/case_info" + str(case_id) + ".p", "rb"))
    files_names = []
    file_path = "../../SimData/TrainingData"
    for _, _, files in os.walk(file_path):
        files_names.extend(files)
    files_names.sort()

    # Construct boundary idx
    full_idx = np.arange(case_info['mesh_num_vert'])
    culled_idx = np.delete(full_idx, case_info['dirichlet'])
    ori_boundary_pt_idx = case_info['boundary'][0]
    ori_boundary_pt_idx = np.sort(np.fromiter(ori_boundary_pt_idx, int))

    hash_table = {}
    for i in range(len(culled_idx)):
        hash_table[culled_idx[i]] = i
    global_culled_boundary_pt_list = []
    local_culled_boundary_pt_list = []

    for i in range(len(ori_boundary_pt_idx)):
        if hash_table.get(ori_boundary_pt_idx[i]) is not None:
            global_culled_boundary_pt_list.append(hash_table[ori_boundary_pt_idx[i]])
            local_culled_boundary_pt_list.append(ori_boundary_pt_idx[i])
    global_culled_boundary_pt_idx = np.asarray(global_culled_boundary_pt_list)
    local_culled_boundary_pt_idx = np.asarray(local_culled_boundary_pt_list)

    # Calculate global nn info
    cluster, _, cluster_parent, belongs, _ = load_cluster("../..", case_id, cluster_num, case_info['mesh_num_vert'])
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

    culled_cluster_num = unique_num
    graph_node_num = len(culled_idx)
    culled_cluster = cluster

    # Read file data
    # Divide workloads:
    cpu_cnt = os.cpu_count()
    print("cpu core account: ", cpu_cnt)

    files_cnt = len(files_names)
    min_files_per_proc_cnt = files_cnt // cpu_cnt
    max_files_per_proc_cnt = min_files_per_proc_cnt + 1
    workload_list = []
    procced_files_cnt = 0
    for i in range(cpu_cnt):
        # [[proc1 first file idx, proc1 last file idx] ... []]
        files_per_proc_cnt = max_files_per_proc_cnt
        if min_files_per_proc_cnt * (cpu_cnt - i) == (files_cnt - procced_files_cnt):
            files_per_proc_cnt = min_files_per_proc_cnt
        cur_proc_workload = [procced_files_cnt, procced_files_cnt + files_per_proc_cnt - 1]
        procced_files_cnt += files_per_proc_cnt
        workload_list.append(cur_proc_workload)
    print("After determining the workload.")

    # Call multi-processing func to load samples:
    pool = mp.Pool()
    proc_list = []
    local_sample_list = []
    global_sample_list = []
    for i in range(cpu_cnt):
        start_idx = workload_list[i][0]
        end_idx = workload_list[i][1]
        loaded_files_names = files_names[start_idx:end_idx+1]
        proc_list.append(pool.apply_async(func=mp_load_data,
                                          args=(start_idx, end_idx, i, edge_index, file_path, loaded_files_names,
                                                local_culled_boundary_pt_idx, culled_idx,)))
    print("After assigning the workload.")

    # Get multi-processing res:
    for i in range(cpu_cnt):
        sub_sample_list = proc_list[i].get()
        local_sample_list.extend(sub_sample_list['local_sample_list'])
        global_sample_list.extend(sub_sample_list['global_sample_list'])

    pool.close()
    pool.join()

    print("Local sample length:", len(local_sample_list))
    print("Global sample length:", len(global_sample_list))

    train_info = {
        "case_name": case_info['case_name'],
        "local_sample_list": local_sample_list,
        "global_sample_list": global_sample_list,
        "local_culled_boundary_points_id": local_culled_boundary_pt_idx,
        "global_culled_boundary_points_id": global_culled_boundary_pt_idx,
        "culled_cluster_num": culled_cluster_num,
        "graph_node_num": graph_node_num,
        "edge_idx": edge_index,
        "culled_cluster": culled_cluster,
        "culled_idx": culled_idx,
        "files_num": len(files_names),
        "cluster_parent": cluster_parent,
        "belongs": belongs
    }

    pickle.dump(train_info, open("../../SimData/TrainingDataPickle/train_info" + str(case_id) + "_" +
                                 str(cluster_num) + "_" + additional_note + ".p", "wb"))

    print("Data process time:", time.time() - load_data_t_start)

