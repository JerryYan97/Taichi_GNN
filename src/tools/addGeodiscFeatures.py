import pickle
import sys
import os
import numpy as np
from numpy import linalg as LA
import taichi as ti
import tina
from src.Utils.utils_visualization import update_boundary_mesh_np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.reader import read

# adj_mat: num_vert x max_adj_v_cnt -- Record adj vert idx with the vert with row idx.
# adj_len: Similarly, it is used to store the length.
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


def gen_dist_arr(adj_mat, adj_len, fixed_pt_idx, scale):
    vert_num = adj_mat.shape[0]
    max_adj_cnt = adj_mat.shape[1]
    dist_arr = 100000.0 * np.ones(vert_num, dtype=float)
    dist_arr[fixed_pt_idx] = 0.0
    spt_set = np.zeros(vert_num, dtype=int)  # 0 -- false, 1 -- true;
    # Start BSF algorithm
    for i in range(vert_num):
        # Pick a vert u which is not there in sptSet and has minimum distance value.
        u = -1
        min_dist = 100000.0
        for j in range(vert_num):
            if spt_set[j] == 0 and dist_arr[j] <= min_dist:
                u = j
                min_dist = dist_arr[j]
        # Include u to sptSet.
        spt_set[u] = 1
        # Update distance value of all adjacent vertices of u.
        for j in range(max_adj_cnt):
            adj_idx = adj_mat[u, j]
            adj_idx_len = adj_len[u, j]
            if adj_idx == -1:
                break
            sum_dist = dist_arr[u] + adj_idx_len
            if sum_dist < dist_arr[adj_idx]:
                dist_arr[adj_idx] = sum_dist
    return dist_arr

# Geodesic.
if __name__ == '__main__':
    fixed_state_spec = "RightHandFixed"
    case_id = 1009
    case_info = read(case_id)

    ti.init(ti.gpu)
    scene = tina.Scene()

    # Construct the SPT tree between fixed points and dynamic points
    fixed_pt_idx = np.array(case_info["dirichlet"])
    mesh = case_info['mesh']
    scale = LA.norm(mesh.bbox[0] - mesh.bbox[1])
    adj_mat, adj_len = get_mesh_map(mesh)
    dist_arr = gen_dist_arr(adj_mat, adj_len, fixed_pt_idx, scale)

    scene_info = {}
    scene_info['scene'] = tina.Scene(culling=False, clipping=True)
    scene_info['tina_mesh'] = tina.SimpleMesh()
    scene_info['model'] = tina.MeshTransform(scene_info['tina_mesh'])
    scene_info['scene'].add_object(scene_info['model'])
    scene_info['boundary_pos'] = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)

    particles = tina.SimpleParticles()
    pars_mat = tina.BlinnPhong()
    particle_radii = 0.01
    scene_info['scene'].add_object(particles, pars_mat)
    particles.set_particles(mesh.vertices)
    particles.set_particle_radii(np.full(mesh.num_vertices, particle_radii))
    par_col = np.full((1, 3), -1.0, dtype=float)
    par_col[0][0] = 1.0
    par_col[0][1] = 1.0
    par_col[0][2] = 1.0
    sim_par_col = np.repeat(par_col, mesh.num_vertices, axis=0)

    col_offset = -dist_arr/scale
    col_offset_mat = np.full((mesh.num_vertices, 3), 0.0, dtype=float)
    col_offset_mat[:, 1] += col_offset
    col_offset_mat[:, 2] += col_offset
    sim_par_col += col_offset_mat

    particles.set_particle_colors(sim_par_col)

    gui = ti.GUI('Test Visualizer')

    update_boundary_mesh_np(mesh.vertices, scene_info['boundary_pos'], case_info)
    pickle.dump(dist_arr, open("../../MeshModels/MeshInfo/geodesic_" + str(case_id) + "_" + fixed_state_spec + ".p", "wb"))
    while True:
        scene_info['scene'].input(gui)
        scene_info['tina_mesh'].set_face_verts(scene_info['boundary_pos'])
        scene_info['scene'].render()
        gui.set_image(scene_info['scene'].img)
        gui.show()

