import pickle
import sys
import os
import torch
import numpy as np
import time
import multiprocessing as mp


# Each sample in it is in normal PyTorch format.
# filepath is used to determine whether read files from training data folder or testing data folder.
# It won't append global feature vector to each sample, because it will blow up the RAM.
def mp_load_local_data(start_file_idx, end_file_idx, proc_idx, filepath, files, culled_bd_idx, culled_idx):
    sample_list = []
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

        sample = {'x_frame': x_frame_data,
                  'y_frame': y_frame_data,
                  'x_frame_full': x_frame_full_data,
                  'y_frame_full': y_frame_full_data,
                  'filename': files[idx]}
        sample_list.append(sample)

    print("proc", proc_idx, " finish works.")
    print("First file:", files[0], ". Last file:", files[len(files) - 1])
    return sample_list


if __name__ == '__main__':
    load_data_t_start = time.time()
    # Read file names
    case_id = 1011
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
    sample_list = []
    for i in range(cpu_cnt):
        start_idx = workload_list[i][0]
        end_idx = workload_list[i][1]
        loaded_files_names = files_names[start_idx:end_idx+1]
        proc_list.append(pool.apply_async(func=mp_load_local_data,
                                          args=(start_idx, end_idx, i, file_path, loaded_files_names,
                                                local_culled_boundary_pt_idx, culled_idx,)))
    print("After assigning the workload.")

    # Get multi-processing res:
    for i in range(cpu_cnt):
        sub_sample_list = proc_list[i].get()
        sample_list.extend(sub_sample_list)

    pool.close()
    pool.join()

    print("Sample length:", len(sample_list))

    train_info = {
        "local_sample_list": sample_list,
        "local_culled_boundary_pt_idx": local_culled_boundary_pt_idx,
        "global_culled_boundary_pt_idx": global_culled_boundary_pt_idx
    }

    pickle.dump(train_info, open("../../SimData/TrainingDataPickle/train_info" + str(case_id) + "_full_data.p", "wb"))

    print("Data process time:", time.time() - load_data_t_start)

