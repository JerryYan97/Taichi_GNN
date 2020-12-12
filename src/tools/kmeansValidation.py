import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.utils_gcn import K_means
from Utils.reader import read


if __name__ == "__main__":
    k = 2
    mesh, dirichlet, mesh_scale, mesh_offset = read(1)
    _, child_list, parent_list, belonging = K_means(mesh, k)
    color_tab = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:olive', 'tab:gray', 'tab:cyan', 'tab:pink',
                 'tab:red', 'tab:brown']
    for i in range(k):
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