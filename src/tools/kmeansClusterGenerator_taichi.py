import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
import sys
import os
from numpy import linalg as LA
from scipy.spatial import KDTree
from collections import Counter
import taichi as ti
import tina
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.reader import read

# Owing to the fixed color panel, it now just has only tests clusters num that is 10 and under 10.
cluster_num = 8
# Case 1002 doesn't work because it's particles' num is less than the clusters' num.
test_case = 1011

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


@ti.data_oriented
class KMeansTaichiHelper:
    def __init__(self, n_vert, n_cluster, adj_mat, adj_len, v_pos):
        self._kdTree = KDTree(v_pos)
        self._min_dist_set = ti.field(ti.i64, shape=(n_cluster, n_vert))
        self._adj_mat = ti.field(ti.i64, shape=(adj_mat.shape[0], adj_mat.shape[1]))
        self._adj_len = ti.field(ti.f64, shape=(adj_mat.shape[0], adj_mat.shape[1]))
        self._v_pos = ti.Vector.field(3, ti.f64, v_pos.shape[0])
        self._cluster_pt_cnt = ti.field(ti.i64, shape=(n_cluster,))
        self._cluster_num = n_cluster
        self._vert_num = n_vert
        self._adj_mat_col_num = adj_mat.shape[1]
        self._adj_mat.from_numpy(adj_mat)
        self._adj_len.from_numpy(adj_len)
        self._v_pos.from_numpy(v_pos)

    @ti.func
    def adj_mat(self, u, v) -> ti.f64:
        uv_len = 0.0
        # Determine whether u and v are connected
        for i in range(self._adj_mat_col_num):
            adj_idx = self._adj_mat[u, i]
            if adj_idx == -1:
                # U and V is not adj.
                break
            elif adj_idx == v:
                # Find the length between them.
                uv_len = self._adj_len[u, i]
                break
        return uv_len

    @ti.kernel
    def generate_spt_list(self,
                          spt_list: ti.ext_arr(),
                          src_list: ti.ext_arr()):
        # Init tmp variables
        for i in range(self._vert_num):
            for k in range(self._cluster_num):
                spt_list[k, i] = 100000.0  # Init a large enough number.
                self._min_dist_set[k, i] = 0  # 0 -- false, 1 -- true.
        # BSF algorithm starts
        for cluster_idx in range(self._cluster_num):
            spt_list[cluster_idx, int(src_list[cluster_idx])] = 0.0
            for vert_idx in range(self._vert_num):
                # minimum distance vertex that is not processed
                min_dist = 100000.0
                u = -1
                for v in range(self._vert_num):
                    if spt_list[cluster_idx, v] < min_dist and self._min_dist_set[cluster_idx, v] == 0:
                        min_dist = spt_list[cluster_idx, v]
                        u = v
                # put minimum distance vertex in shortest tree
                self._min_dist_set[cluster_idx, u] = 1
                # Update dist value of the adjacent vertices
                for v in range(self._vert_num):
                    if self.adj_mat(u, v) > 0.0 and self._min_dist_set[cluster_idx, v] == 0 and spt_list[cluster_idx, v] > spt_list[cluster_idx, u] + self.adj_mat(u, v):
                        spt_list[cluster_idx, v] = spt_list[cluster_idx, u] + self.adj_mat(u, v)

    @ti.kernel
    def generate_belongs(self,
                         spt_list: ti.ext_arr(),
                         belong_list: ti.ext_arr(),
                         belong_len_list: ti.ext_arr()):
        for v in range(self._vert_num):
            min_dis = 100000.0
            cluster_id = -1
            for cluster_idx in range(self._cluster_num):
                dis = spt_list[cluster_idx, v]
                if dis < min_dis:
                    cluster_id = cluster_idx
                    min_dis = dis
            belong_list[v] = cluster_id
            belong_len_list[v] = min_dis

    @ti.kernel
    def calculate_center(self, belong_list: ti.ext_arr(), center_pos: ti.ext_arr()):
        for cluster_idx in range(self._cluster_num):
            self._cluster_pt_cnt[cluster_idx] = 0
        for v in range(self._vert_num):
            cluster_idx = int(belong_list[v])
            center_pos[cluster_idx, 0] += self._v_pos[v][0]
            center_pos[cluster_idx, 1] += self._v_pos[v][1]
            center_pos[cluster_idx, 2] += self._v_pos[v][2]
            self._cluster_pt_cnt[cluster_idx] += 1
        for cluster_idx in range(self._cluster_num):
            center_pos[cluster_idx, 0] /= self._cluster_pt_cnt[cluster_idx]
            center_pos[cluster_idx, 1] /= self._cluster_pt_cnt[cluster_idx]
            center_pos[cluster_idx, 2] /= self._cluster_pt_cnt[cluster_idx]

    @ti.kernel
    def calculate_center_change(self,
                                parent_list_new: ti.ext_arr(),
                                parent_list_old: ti.ext_arr()) -> ti.f64:
        change_amount = 0.0
        for cluster_idx in range(self._cluster_num):
            old_cluster_center_idx = int(parent_list_old[cluster_idx])
            new_cluster_center_idx = int(parent_list_new[cluster_idx])
            change_amount += (self._v_pos[old_cluster_center_idx] - self._v_pos[new_cluster_center_idx]).norm()
        return change_amount

    def update_center(self, parent_list, belong_list):
        parent_list_last = parent_list
        center_pos = np.zeros(shape=(self._cluster_num, 3), dtype=np.float)
        self.calculate_center(belong_list, center_pos)
        dd, ii = self._kdTree.query(center_pos)

        # Check duplicate cluster center:
        # https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
        tmp = [item for item, count in Counter(ii).items() if count > 1]
        if len(tmp) != 0:
            raise Exception("There is a duplicate cluster center.")

        parent_list_new = np.asarray(ii, dtype=np.int)
        change_amount = self.calculate_center_change(parent_list_new, parent_list_last)
        return parent_list_new, change_amount


def get_mesh_map(mesh):
    mesh.enable_connectivity()
    max_adj_v_cnt = -1
    adj_v_cnt_sum = 0
    for p in range(mesh.num_vertices):
        adj_v = mesh.get_vertex_adjacent_vertices(p)
        adj_v_cnt_sum += adj_v.shape[0]
        if adj_v.shape[0] > max_adj_v_cnt:
            max_adj_v_cnt = adj_v.shape[0]
    adj_v_cnt_avg = float(adj_v_cnt_sum) / float(mesh.num_vertices)
    print("avg adj vertex count:", adj_v_cnt_avg)
    print("max adj vertex count:", max_adj_v_cnt)
    adj_mat = -np.ones((mesh.num_vertices, max_adj_v_cnt), dtype=np.int)
    adj_len = -np.ones((mesh.num_vertices, max_adj_v_cnt), dtype=np.float)
    for p in range(mesh.num_vertices):
        adj_v = mesh.get_vertex_adjacent_vertices(p)
        for i in range(adj_v.shape[0]):
            adj_mat[p, i] = adj_v[i]
            adj_len[p, i] = LA.norm(mesh.vertices[p] - mesh.vertices[adj_v[i]])
    print("get_vert_adj_mat finishes.")
    return adj_mat, adj_len


def K_means_taichi(mesh, k):
    if k > mesh.num_vertices:
        raise Exception("k should be less than mesh's vertices num.")

    if k < os.cpu_count():
        raise Exception("Currently it doesn't support clusters num less than cpu cores num.")

    # Two adjustable parameters to control the convergence:
    max_itr = 6
    # If the change amount is less than (1 - convergence_rate) * bbox diagonal length, then it should be considered as
    # convergence.
    convergence_rate = 0.9
    bbox_diag_len = LA.norm(mesh.bbox[0] - mesh.bbox[1])

    parent_list = []
    step = mesh.num_vertices // k
    for i in range(k):
        parent_list.append(i * step)
    parent_list = np.asarray(parent_list, dtype=np.int)
    parent_list_last = parent_list

    belonging = np.arange(mesh.num_vertices, dtype=np.int)
    belonging_len = np.arange(mesh.num_vertices, dtype=np.float)

    adj_mat, adj_len = get_mesh_map(mesh)

    spt_list = np.zeros((cluster_num, mesh.num_vertices), dtype=np.float)
    kmeans_helper = KMeansTaichiHelper(mesh.num_vertices, k, adj_mat, adj_len, mesh.vertices)

    for itr in range(max_itr):
        # assignment each point to its nearest center
        kmeans_helper.generate_spt_list(spt_list, parent_list)
        kmeans_helper.generate_belongs(spt_list, belonging, belonging_len)

        # recalculate the center for each cluster
        parent_list_last = parent_list
        parent_list, change_amount = kmeans_helper.update_center(parent_list, belonging)
        print("Total center pos change amount (Less is better):", change_amount)

        if (1.0 - convergence_rate) * bbox_diag_len > change_amount:
            break

    return parent_list_last, belonging, belonging_len


if __name__ == "__main__":
    case_info = read(test_case)
    mesh = case_info['mesh']
    dirichlet = case_info['dirichlet']
    mesh_scale = case_info['mesh_scale']
    mesh_offset = case_info['mesh_offset']
    dim = case_info['dim']

    if dim == 2 or mesh.num_vertices < 100:
        raise Exception("Please use the kmeansClusterGenerator.py instead. That is faster than this in small scale.")

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

    time_start = time.time()
    parent_list, belonging, belonging_len = K_means_taichi(mesh, cluster_num)
    time_end = time.time()
    print("Kmeans execute time duration:", time_end-time_start, 's')

    # Save the cluster
    if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/SavedClusters"):
        os.makedirs("Saved_Cluster")
    cluster = np.zeros(len(mesh.vertices) * 2 + cluster_num + 1, dtype=np.float)
    # Save clusters:
    # NOTE: The new version of the cluster format is different from the original one.
    # Section 1: Number of clusters in this file.
    # Section 2: Belongs of all the vertices. For each element, the value spans from 0 -- number of cluster
    # Section 3: The length of them from center.
    # Section 4: Parent list.
    cluster[0] = cluster_num
    cluster[1:1 + mesh.num_vertices] = belonging
    cluster[1 + mesh.num_vertices:1 + 2 * mesh.num_vertices] = belonging_len
    cluster[1 + 2 * mesh.num_vertices:1 + 2 * mesh.num_vertices + cluster_num] = parent_list
    np.savetxt(os.path.dirname(os.path.abspath(__file__)) +
               "/../../MeshModels/SavedClusters/" + f"test_case{test_case}_c{cluster_num}_cluster.csv",
               cluster, delimiter=',')

    gui = ti.GUI('kmeans visualization')
    pars.set_particles(mesh.vertices)
    pars.set_particle_radii(np.full(mesh.num_vertices, 0.05))
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
