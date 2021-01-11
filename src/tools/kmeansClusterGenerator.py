import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
# from Utils.utils_gcn import K_means, K_means_multiprocess, K_means_taichi
from Utils.utils_gcn import *
from Utils.reader import read

# Owing to the fixed color panel, it now just has only tests clusters num that is 10 and under 10.
cluster_num = 10
# Case 1002 doesn't work because it's particles' num is less than the clusters' num.
test_case = 1005


# Optimization record:
# Before multiprocessing optimization:
# case 1: 21.836742877960205 s
# case 2: 1660.8389155864716 s
# case 3: 91.31829476356506 s
# case 4: 463.22716546058655 s
# After multiprocessing optimization:
# case 1: 3.13779354095459 s
# case 2: 245.91074132919312 s
# case 3: 12.78972315788269 s
# case 4: 76.89776110649109 s
# Multiprocessing + Dijkstra process optimization:
# case 1: 0.667464017868042 s
# case 2: 35.071603536605835 s
# case 3: 2.235482692718506 s
# case 4: 10.509188652038574 s
# case 1001: 8.820295810699463 s
# case 1002: None
# case 1003: Timeout
# case 1004: Timeout
# Multiprocessing 2 + Taichi optimization:
# case 1001: ~13.113672256469727s
# case 1004: 2141.33443069458 s
# case 1005: 3529.759060382843 s
# Opt3:
# case 1: 0.48258471488952637 s
# case 2: 6.196524143218994 s
# case 1001: 3.125687599182129 s
# case 1004: 384.59628558158875 s
# case 1005:

def rgb_range01(rgb_np):
    return rgb_np / 255.0


if __name__ == "__main__":
    case_info = read(test_case)
    mesh = case_info['mesh']
    dirichlet = case_info['dirichlet']
    mesh_scale = case_info['mesh_scale']
    mesh_offset = case_info['mesh_offset']
    dim = case_info['dim']

    time_start = time.time()
    _, child_list, parent_list, belonging = K_means_multiprocess(mesh, cluster_num)
    time_end = time.time()
    print("Kmeans execute time duration:", time_end-time_start, 's')

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

    if dim == 2:
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
    elif dim == 3:
        import tina
        import taichi as ti

        ti.init(ti.gpu)

        scene = tina.Scene()

        # Init particles info
        particles_list = []
        label_color_list = np.array([rgb_range01(np.array([255.0, 66.0, 66.0])),
                                     rgb_range01(np.array([245.0, 167.0, 66.0])),
                                     rgb_range01(np.array([245.0, 239.0, 66.0])),
                                     rgb_range01(np.array([129.0, 245.0, 66.0])),
                                     rgb_range01(np.array([66.0, 245.0, 132.0])),
                                     rgb_range01(np.array([66.0, 245.0, 218.0])),
                                     rgb_range01(np.array([66.0, 197.0, 245.0])),
                                     rgb_range01(np.array([66.0, 114.0, 245.0])),
                                     rgb_range01(np.array([144.0, 66.0, 245.0])),
                                     rgb_range01(np.array([242.0, 66.0, 245.0]))])

        pars = tina.SimpleParticles()
        material = tina.BlinnPhong()
        scene.add_object(pars, material)

        gui = ti.GUI('kmeans visualization')

        pars.set_particles(mesh.vertices)
        pars.set_particle_radii(np.full(mesh.num_vertices, 0.01))
        # Label particles color
        particles_color = np.full((mesh.num_vertices, 3), -1.0, dtype=float)
        np_child_list = np.asarray(child_list)
        for i in range(cluster_num):
            pos_idx = np.asarray(np.asarray(belonging) == parent_list[i]).nonzero()[0]
            particles_color[np_child_list[pos_idx]] = label_color_list[i]
            particles_color[parent_list[i]] = label_color_list[i]
        pars.set_particle_colors(particles_color)
        while gui.running:
            scene.input(gui)
            scene.render()
            gui.set_image(scene.img)
            gui.show()