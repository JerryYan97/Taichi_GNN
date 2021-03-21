import os
import numpy as np

if __name__ == '__main__':
    # Read in target full data names:
    tar_data_path = "./tmpFullData"
    tar_data_name_list = []
    for _, _, files in os.walk(tar_data_path):
        tar_data_name_list.extend(files)
    for i in range(len(tar_data_name_list)):
        tmp_str = tar_data_name_list[i][0:20] + tar_data_name_list[i][27:]
        print(tmp_str)
        os.rename(tar_data_path + "/" + tar_data_name_list[i], tar_data_path + "/" + tmp_str)

