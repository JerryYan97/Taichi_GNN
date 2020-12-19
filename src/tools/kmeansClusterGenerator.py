import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.utils_gcn import K_means
from Utils.reader import read

cluster_num = 10
test_case = 4

if __name__ == "__main__":
    case_info = read(test_case)
    mesh = case_info['mesh']
    dirichlet = case_info['dirichlet']
    mesh_scale = case_info['mesh_scale']
    mesh_offset = case_info['mesh_offset']

    _, child_list, parent_list, belonging = K_means(mesh, cluster_num)
    color_tab = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:olive', 'tab:gray', 'tab:cyan', 'tab:pink',
                 'tab:red', 'tab:brown']

    # Save the cluster
    if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/SavedClusters"):
        os.makedirs("Saved_Cluster")
    cluster = np.zeros(len(mesh.vertices) + 1, dtype=int)
    for i in parent_list:
        cluster[i] = i
    for i in range(len(child_list)):
        cluster[child_list[i]] = belonging[i]
    cluster[len(mesh.vertices)] = cluster_num
    np.savetxt(os.path.dirname(os.path.abspath(__file__)) +
               "/../../MeshModels/SavedClusters/" + f"test_case{test_case}_cluster.csv",
               cluster, delimiter=',', fmt='%d')

    for i in range(cluster_num):
        posidx = np.where(np.asarray(belonging) == parent_list[i])[0]
        pos = mesh.vertices[np.asarray(child_list)[posidx]]
        x = [item[0] for item in pos]
        x.append(mesh.vertices[parent_list[i], 0])
        y = [item[1] for item in pos]
        y.append(mesh.vertices[parent_list[i], 1])
        plt.scatter(x, y, label="stars", color=color_tab[i], marker="*", s=30)

    plt.xlabel('x - axis')  # x-axis label
    plt.ylabel('y - axis')  # frequency label
    plt.title('K-means result plot!')  # plot title
    plt.legend()  # showing legend
    plt.show()  # function to show the plot
    # plt.savefig('kmeans_result.png')