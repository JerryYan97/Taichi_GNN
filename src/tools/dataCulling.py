import os
import numpy as np

# This tool is used to cull data to determine the useful features in training data.
if __name__ == '__main__':
    # Read in target full data names:
    tar_data_path = "./tmpFullData"
    tar_data_name_list = []
    for _, _, files in os.walk(tar_data_path):
        tar_data_name_list.extend(files)

    save_data_path = "./tmpCulledData"
    os.makedirs(save_data_path, exist_ok=True)
    for root, dirs, files in os.walk(save_data_path):
        for name in files:
            os.remove(os.path.join(root, name))

    for i in range(len(tar_data_name_list)):
        tar_data = np.genfromtxt(tar_data_path + "/" + tar_data_name_list[i], delimiter=',')
        # Delete target data column:
        # Cull: Acc (15,16,17), boundary label (24), dt (25)
        culled_data = np.delete(tar_data, [15, 16, 17, 24, 25], 1)
        np.savetxt(save_data_path + "/" + tar_data_name_list[i], culled_data, delimiter=',')
        print("Finished_data_cull:", i, "/", len(tar_data_name_list))

