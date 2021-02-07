import sys, os, time
import numpy as np
import multiprocessing as mp
import os
from collections import defaultdict
import random
import time
import struct
import h5py

# using method 
datapath = "../../SimData/h5"

os.makedirs(datapath + "../h52csv", exist_ok=True)
os.makedirs(datapath + "../csv2h5", exist_ok=True)

def ToBinary(workload_list, proc_idx, filepath, files):
    for idx in range(workload_list[proc_idx][0], workload_list[proc_idx][1] + 1):
        fperframe = np.genfromtxt(filepath + "/" + files[idx], delimiter=',')
        hf = h5py.File(filepath + "/../csv2h5/" + files[idx][:-4]+".h5", 'w')
        hf.create_dataset('dataset_1', data=fperframe)
        n1 = hf.get('dataset_1')
        n1 = np.array(n1)
        hf.close()

def FromBinary(workload_list, proc_idx, filepath, files):
    for idx in range(workload_list[proc_idx][0], workload_list[proc_idx][1] + 1):
        hf = h5py.File(filepath + "/" + files[idx][:-3]+".h5", 'r')
        n1 = hf.get('dataset_1')
        n1 = np.array(n1)
        np.savetxt(filepath + "/../h52csv/" + files[idx][:-3] + ".csv", n1)
        hf.close()

if __name__ == "__main__":
    pool = mp.Pool()
    sample_list = []
    cpu_cnt = os.cpu_count()
    files_cnt = len([name for name in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, name))])
    print("cpu core account: ", cpu_cnt, "data size: ", files_cnt)
    files_per_proc_cnt = files_cnt // cpu_cnt
    workload_list = []
    proc_list = []
    for i in range(cpu_cnt):
        cur_proc_workload = [i * files_per_proc_cnt, (i + 1) * files_per_proc_cnt - 1]
        if i == cpu_cnt - 1:
            cur_proc_workload[1] = files_cnt - 1
        workload_list.append(cur_proc_workload)
    # print(workload_list)
    # Call multi-processing func:
    for _, _, files in os.walk(datapath):
        files.extend(files)
    for i in range(cpu_cnt):
        pool.apply_async(func=FromBinary, args=(workload_list, i, datapath, files,))
    pool.close()
    pool.join()