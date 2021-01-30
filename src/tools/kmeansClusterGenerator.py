import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
# from Utils.utils_gcn import K_means, K_means_multiprocess, K_means_taichi
from Utils.utils_gcn import *
from Utils.reader import read

# Owing to the fixed color panel, it now just has only tests clusters num that is 10 and under 10.
cluster_num = 8
# Case 1002 doesn't work because it's particles' num is less than the clusters' num.
test_case = 2


def rgb_range01(rgb_np):
    return rgb_np / 255.0


# https://www.iquilezles.org/www/articles/palettes/palettes.htm
def color_palettes(t):
    if t > 1.0 or t < 0.0:
        raise Exception("Input should be [0, 1].")
    a = np.array([0.5, 0.5, 0.5])
    b = np.array([0.5, 0.5, 0.5])
    c = np.array([1.0, 1.0, 1.0])
    d = np.array([0.0, 0.33, 0.67])
    return a + b * np.cos(2.0 * np.pi * (c * t + d))


if __name__ == "__main__":
    case_info = read(test_case)
    mesh = case_info['mesh']
    dirichlet = case_info['dirichlet']
    mesh_scale = case_info['mesh_scale']
    mesh_offset = case_info['mesh_offset']
    dim = case_info['dim']

    time_start = time.time()
    parent_list, belonging, belonging_len = K_means_multiprocess2(mesh, cluster_num)
    time_end = time.time()
    print("Kmeans execute time duration:", time_end-time_start, 's')

    # Save the cluster
    if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/SavedClusters"):
        os.makedirs("Saved_Cluster")
    cluster = np.zeros(len(mesh.vertices) * 2 + cluster_num + 1, dtype=np.float)
    # Save clusters:
    # NOTE: The new version of the cluster format is different from the original one.
    # Line 1: number of clusters in this file.
    # Line 2 -- 2 + n_vert: belongs of all the vertices. For each element, the value spans from 0 -- number of cluster
    # Line 2 + n_vert + 1 -- 2 + 2 * n_vert + 1: The length of them from center.
    cluster[0] = cluster_num
    cluster[1:1 + mesh.num_vertices] = belonging
    cluster[1 + mesh.num_vertices:1 + 2 * mesh.num_vertices] = belonging_len
    cluster[1 + 2 * mesh.num_vertices:1 + 2 * mesh.num_vertices + cluster_num] = parent_list
    np.savetxt(os.path.dirname(os.path.abspath(__file__)) +
               "/../../MeshModels/SavedClusters/" + f"test_case{test_case}_c{cluster_num}_cluster.csv",
               cluster, delimiter=',')

    if dim == 2:
        for i in range(cluster_num):
            posidx = np.where(np.asarray(belonging) == i)[0]
            pos = mesh.vertices[posidx]
            x = [item[0] for item in pos]
            x.append(mesh.vertices[parent_list[i], 0])
            y = [item[1] for item in pos]
            y.append(mesh.vertices[parent_list[i], 1])
            plt.scatter(x, y, label="stars", marker="*", s=30)

        plt.xlabel('x - axis')  # x-axis label
        plt.ylabel('y - axis')  # frequency label
        plt.title('K-means result plot!')  # plot title
        plt.legend()  # showing legend
        plt.show()  # function to show the plot
        # plt.savefig('kmeans_result.png')
    elif dim == 3:
        import tina
        import taichi as ti

        ti.init(ti.gpu)

        scene = tina.Scene()

        # Init particles info
        particles_list = []
        label_color_list = np.array(color_palettes(0.0))
        for i in range(cluster_num - 1):
            label_color_list = np.vstack((label_color_list, color_palettes(float(i + 1) / float(cluster_num))))

        pars = tina.SimpleParticles()
        material = tina.BlinnPhong()
        scene.add_object(pars, material)

        gui = ti.GUI('kmeans visualization')

        pars.set_particles(mesh.vertices)
        pars.set_particle_radii(np.full(mesh.num_vertices, 0.1))
        # Label particles color
        particles_color = np.full((mesh.num_vertices, 3), -1.0, dtype=float)
        for i in range(mesh.num_vertices):
            belong_cluster = belonging[i]
            particles_color[i] = label_color_list[belong_cluster]
        pars.set_particle_colors(particles_color)
        while gui.running:
            scene.input(gui)
            scene.render()
            gui.set_image(scene.img)
            gui.show()