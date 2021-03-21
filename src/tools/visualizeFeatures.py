import sys
import os
import numpy as np
import taichi as ti
import tina
from src.Utils.utils_visualization import update_boundary_mesh_np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.reader import read


class SceneHelper():
    def __init__(self):
        self.feature_file_path = "../../SimData/TrainingData"
        self._feature_files_list = []
        for _, _, files in os.walk(self.feature_file_path):
            self._feature_files_list.extend(files)
        self._feature_files_list.sort()
        print("Feature files:\n", self._feature_files_list)
        self.file_data = np.genfromtxt(self.feature_file_path + "/" + self._feature_files_list[0], delimiter=',')
        self.cur_feat_id = 0  # 0 -- Geodesic, 1 -- Potential, 2 -- Digression;
        cur_feat_data = self.file_data[:, 26]
        par_col = np.full((1, 3), -1.0, dtype=float)
        par_col[0][0] = 1.0
        par_col[0][1] = 1.0
        par_col[0][2] = 1.0
        self.base_par_col = np.repeat(par_col, mesh.num_vertices, axis=0)

        col_offset = -cur_feat_data  # Geodesic
        col_offset_mat = np.full((mesh.num_vertices, 3), 0.0, dtype=float)
        col_offset_mat[:, 1] += col_offset
        col_offset_mat[:, 2] += col_offset

        self.cur_par_col = self.base_par_col + col_offset_mat

    def change_feature(self, particles):
        self.cur_feat_id = (self.cur_feat_id + 1) % 3
        col_offset_mat = np.full((mesh.num_vertices, 3), 0.0, dtype=float)
        if self.cur_feat_id == 0:
            # Geodesic
            cur_feat_data = np.copy(self.file_data[:, 26])
            col_offset = -cur_feat_data
            col_offset_mat[:, 1] += col_offset
            col_offset_mat[:, 2] += col_offset
        elif self.cur_feat_id == 1:
            # Potential
            cur_feat_data = np.copy(self.file_data[:, 27])
            col_offset = -cur_feat_data
            col_offset_mat[:, 0] += col_offset
            col_offset_mat[:, 2] += col_offset
        elif self.cur_feat_id == 2:
            # Digression
            dig_feat_data = np.copy(self.file_data[:, 28])
            col_offset = dig_feat_data
            fixed_pt_idx = np.where(col_offset < 0.0)
            col_offset[fixed_pt_idx] = 0.0
            col_offset /= -col_offset.max()
            col_offset_mat[:, 0] += col_offset
            col_offset_mat[:, 1] += col_offset
        self.cur_par_col = self.base_par_col + col_offset_mat
        particles.set_particle_colors(self.cur_par_col)
        return self.cur_feat_id


if __name__ == '__main__':
    case_id = 1009
    case_info = read(case_id)
    mesh = case_info['mesh']
    fixed_pt_idx = np.array(case_info["dirichlet"])

    ti.init(ti.gpu)
    scene = tina.Scene()

    helper = SceneHelper()

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
    particles.set_particle_colors(helper.cur_par_col)
    gui = ti.GUI('Features Visualizer')
    update_boundary_mesh_np(mesh.vertices, scene_info['boundary_pos'], case_info)

    key_press_lag = 2
    press_n_counter = 0
    while True:
        scene_info['scene'].input(gui)
        gui.get_event()
        if gui.is_pressed('n'):
            if press_n_counter == key_press_lag:
                press_n_counter = 0
                cur_feat_id = helper.change_feature(particles)
            else:
                press_n_counter += 1

        scene_info['tina_mesh'].set_face_verts(scene_info['boundary_pos'])
        scene_info['scene'].render()
        gui.set_image(scene_info['scene'].img)
        gui.show()

