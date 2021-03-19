import numpy as np
import sys
import os
import pymesh
import tina
import taichi as ti
ti.init(ti.gpu)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.utils_visualization import rotate_matrix_y_axis, update_boundary_pos_np

# Instructions:
# n: next frame; m: previous frame;
# SPACE: Rendering from 0 frame to the last frame;
# r: reset to 0 frame;
# s: jump to targe frame. You need to input the frame id in console;
# SceneHelper: you can adjust mesh transformation and mesh input src in this class.


@ti.kernel
def Find_Similar_Idx(label_array: ti.ext_arr(),
                     train_feature_mat: ti.ext_arr(),
                     tar_feature: ti.ext_arr(),
                     array_len: ti.i32,
                     feature_len: ti.i32):
    for i in range(array_len):
        diff_norm = 0.0
        test_feat_norm = 0.0
        for j in range(feature_len):
            diff_norm += (train_feature_mat[i, j] - tar_feature[j]) ** 2
            test_feat_norm += (tar_feature[j]) ** 2
        diff_percentage = ti.sqrt(diff_norm) / ti.sqrt(test_feat_norm)
        if diff_percentage < 0.05:
            label_array[i] = 1
        else:
            label_array[i] = 0


@ti.kernel
def Find_Similar_Idx_With_Global(label_array: ti.ext_arr(),
                                 train_feature_mat_ng: ti.ext_arr(),
                                 train_feature_g: ti.ext_arr(),
                                 tar_feature_arr_ng: ti.ext_arr(),
                                 tar_feature_g: ti.ext_arr(),
                                 array_len: ti.i32,
                                 ng_feature_len: ti.i32,
                                 g_feature_len: ti.i32,
                                 cluster_num: ti.i32):
    for i in range(array_len):
        diff_norm = 0.0
        test_feat_norm = 0.0
        for j in range(ng_feature_len):
            diff_norm += (train_feature_mat_ng[i, j] - tar_feature_arr_ng[j]) ** 2
            test_feat_norm += (tar_feature_arr_ng[j]) ** 2
        for j in range(g_feature_len):
            for k in range(cluster_num):
                diff_norm += (train_feature_g[k, j] - tar_feature_g[k, j]) ** 2
                test_feat_norm += (tar_feature_g[k, j]) ** 2
        diff_percentage = ti.sqrt(diff_norm) / ti.sqrt(test_feat_norm)
        if diff_percentage < 0.05:
            label_array[i] = 1
        else:
            label_array[i] = 0


def read_mesh(file_path_name):
    mesh = pymesh.load_mesh(file_path_name)
    boundary_pos = np.ndarray(shape=(mesh.num_faces, 3, 3), dtype=np.float)
    update_boundary_pos_np(mesh.vertices, boundary_pos, mesh.faces, mesh.num_faces)
    return boundary_pos


# NOTE: PD-NN visualization is not tested
class SceneHelper():
    def __init__(self):
        # It needs to be changed if we just want to have
        # NOTE: Remember to reconstruct Training PD Anim Seq
        self.PDNN_file_path = "../../SimData/FinalRes/GNNPDAnimSeq"
        self.Train_PD_file_path = "../../SimData/FinalRes/TrainReconstructPDAnimSeq"
        self.PN_file_path = "../../SimData/FinalRes/ReconstructPNAnimSeq"
        self.Train_data_file_path = "../../SimData/TrainingData"
        self.Test_data_file_path = "../../SimData/TestingData"
        self.TrainPreGen_file_path = "../../SimData/TrainPreGenGlobalFeatureVec"
        self.TestPreGen_file_path = "../../SimData/TestPreGenGlobalFeatureVec"

        # self.PD_file_path = "../../SimData/PDAnimSeq"
        # self.PN_file_path = "../../SimData/PNAnimSeq"
        # Init files
        self._PDNN_files_list, self._PD_files_list, self._PN_files_list = [], [], []
        # PD-NN
        for _, _, files in os.walk(self.PDNN_file_path):
            self._PDNN_files_list.extend(files)
        self._PDNN_files_list.sort()
        print("PD-NN Anim files:\n", self._PDNN_files_list)
        # PD
        for _, _, files in os.walk(self.Train_PD_file_path):
            self._PD_files_list.extend(files)
        self._PD_files_list.sort()
        print("PD Anim files:\n", self._PD_files_list)
        # PN
        for _, _, files in os.walk(self.PN_file_path):
            self._PN_files_list.extend(files)
        self._PN_files_list.sort()
        print("PN Anim files:\n", self._PN_files_list)

        self.frame_num = len(self._PD_files_list)

        self.input_trans = [-1.5, 0.0, 0.0]
        self.PD_transform = tina.translate(self.input_trans) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)
        self.PDGNN_transform = tina.translate([0.0, 0.0, 0.0]) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)
        self.PN_transform = tina.translate([1.5, 0.0, 0.0]) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)

        self.PD_mesh = tina.SimpleMesh()
        self.PD_model = tina.MeshTransform(self.PD_mesh)
        self.PD_mat = tina.Lambert(color=[1.0, 0.5, 0.5])

        self.PDGNN_mesh = tina.SimpleMesh()
        self.PDGNN_model = tina.MeshTransform(self.PDGNN_mesh)
        self.PDGNN_mat = tina.Lambert(color=[0.5, 1.0, 0.5])

        self.PN_mesh = tina.SimpleMesh()
        self.PN_model = tina.MeshTransform(self.PN_mesh)
        self.PN_mat = tina.Lambert(color=[0.5, 0.5, 1.0])

        self.selected_particle_idx = 8
        self.particle_radii = 0.01
        self.pars_mat = tina.BlinnPhong()

        # Selected Particles:
        self.select_pars = tina.SimpleParticles()

        # Similar Particles:
        self.similar_pars = tina.SimpleParticles()

        self.scene = tina.Scene(culling=False, clipping=True, res=1024)

        self.scene.add_object(self.PDGNN_model, self.PDGNN_mat)
        self.scene.add_object(self.PD_model, self.PD_mat)
        self.scene.add_object(self.PN_model, self.PN_mat)
        self.scene.add_object(self.select_pars, self.pars_mat)
        self.scene.add_object(self.similar_pars, self.pars_mat)

        # Read Training, testing and GlobalFeature files name:
        self._training_file_list = []
        for _, _, files in os.walk(self.Train_data_file_path):
            self._training_file_list.extend(files)
        self._training_file_list.sort()
        print("Training Data Files:\n", self._training_file_list)

        self._testing_file_list = []
        for _, _, files in os.walk(self.Test_data_file_path):
            self._testing_file_list.extend(files)
        self._testing_file_list.sort()
        print("Testing Data Files:\n", self._testing_file_list)

        self._train_pregen_global_file_list = []
        for _, _, files in os.walk(self.TrainPreGen_file_path):
            self._train_pregen_global_file_list.extend(files)
        self._train_pregen_global_file_list.sort()
        print("Train Pregen Global Files:\n", self._training_file_list)

        self._test_pregen_global_file_list = []
        for _, _, files in os.walk(self.TestPreGen_file_path):
            self._test_pregen_global_file_list.extend(files)
        self._test_pregen_global_file_list.sort()
        print("Test Pregen Global Files:\n", self._testing_file_list)

    def set_transform(self):
        self.PDGNN_model.set_transform(self.PDGNN_transform)
        self.PD_model.set_transform(self.PD_transform)
        self.PN_model.set_transform(self.PN_transform)

    def set_mesh(self, cur_test_frame_id, cur_train_frame_id):
        # Set Mesh
        PD_file_path_name = self.Train_PD_file_path + "/" + self._PD_files_list[cur_train_frame_id]
        PD_boundary_pos = read_mesh(PD_file_path_name)
        self.PD_mesh.set_face_verts(PD_boundary_pos)

        PDGNN_file_path_name = self.PDNN_file_path + "/" + self._PDNN_files_list[cur_test_frame_id]
        PDGNN_boundary_pos = read_mesh(PDGNN_file_path_name)
        self.PDGNN_mesh.set_face_verts(PDGNN_boundary_pos)

        PN_file_path_name = self.PN_file_path + "/" + self._PN_files_list[cur_test_frame_id]
        PN_boundary_pos = read_mesh(PN_file_path_name)
        self.PN_mesh.set_face_verts(PN_boundary_pos)

        # Set Selected Particle
        selected_par_pos = np.full((1, 3), -1.0, dtype=float)
        PDGNN_mesh = pymesh.load_mesh(PDGNN_file_path_name)
        selected_par_pos[0][0] = PDGNN_mesh.vertices[self.selected_particle_idx][0]
        selected_par_pos[0][1] = PDGNN_mesh.vertices[self.selected_particle_idx][1]
        selected_par_pos[0][2] = PDGNN_mesh.vertices[self.selected_particle_idx][2]
        self.select_pars.set_particles(selected_par_pos)
        self.select_pars.set_particle_radii(np.full(1, self.particle_radii))
        par_col = np.full((1, 3), -1.0, dtype=float)
        par_col[0][0] = 1.0
        par_col[0][1] = 0.0
        par_col[0][2] = 0.0
        self.select_pars.set_particle_colors(par_col)

        # Find Similar Particle
        similar_par_pos = np.full((1, 3), -1.0, dtype=float)
        label_array = np.zeros(PDGNN_mesh.num_vertices, dtype=int)
        train_data = np.genfromtxt(self.Train_data_file_path + "/" + self._training_file_list[cur_train_frame_id],
                                   delimiter=',')
        test_data = np.genfromtxt(self.Test_data_file_path + "/" + self._testing_file_list[cur_test_frame_id],
                                  delimiter=',')
        train_global_feat_data = np.genfromtxt(self.TrainPreGen_file_path + "/" + self._train_pregen_global_file_list[cur_train_frame_id],
                                               delimiter=',')
        test_global_feat_data = np.genfromtxt(self.TestPreGen_file_path + "/" + self._test_pregen_global_file_list[cur_test_frame_id],
                                              delimiter=',')

        train_feature_data = np.hstack((train_data[:, 0:3], train_data[:, 6:]))
        test_feature_data = np.hstack((test_data[:, 0:3], test_data[:, 6:]))

        PD_mesh = pymesh.load_mesh(PD_file_path_name)
        # Find_Similar_Idx(label_array,
        #                  train_feature_data,
        #                  test_feature_data[self.selected_particle_idx],
        #                  train_feature_data.shape[0],
        #                  train_feature_data.shape[1])
        Find_Similar_Idx_With_Global(label_array,
                                     train_feature_data,
                                     train_global_feat_data,
                                     test_feature_data[self.selected_particle_idx],
                                     test_global_feat_data,
                                     train_feature_data.shape[0],
                                     train_feature_data.shape[1],
                                     train_global_feat_data.shape[1],
                                     train_global_feat_data.shape[0])
        similar_idx = np.where(label_array == 1)

        # Set Similar Particle
        similar_par_pos = PD_mesh.vertices[similar_idx]
        trans_vec = np.array([[self.input_trans[0], self.input_trans[1], self.input_trans[2]]])
        trans_mat = np.repeat(trans_vec, len(similar_idx[0]), axis=0)
        similar_par_pos += trans_mat
        self.similar_pars.set_particles(similar_par_pos)
        self.similar_pars.set_particle_radii(np.full(len(similar_idx[0]), self.particle_radii))
        sim_par_col = np.repeat(par_col, len(similar_idx[0]), axis=0)
        self.similar_pars.set_particle_colors(sim_par_col)

        return len(similar_idx[0]), train_data.shape[0]

    def select_particle(self, sel_id):
        self.selected_particle_idx = sel_id

    def adjust_radius(self, add_amount):
        self.particle_radii += add_amount

    def start_frame(self):
        file_name_len = len(self._PD_files_list[0])
        file_id_str = self._PD_files_list[0][file_name_len - 9:file_name_len - 4]
        file_id = int(file_id_str)
        return file_id

    def get_particle_num(self):
        PDGNN_file_path_name = self.PDNN_file_path + "/" + self._PDNN_files_list[0]
        mesh = pymesh.load_mesh(PDGNN_file_path_name)
        return mesh.num_vertices


# NOTE: It only works for 3D now
if __name__ == '__main__':
    os.makedirs('results/', exist_ok=True)
    for root, dirs, files in os.walk("results/"):
        for name in files:
            os.remove(os.path.join(root, name))

    # Init scene variables and adjust visualization parameters
    helper = SceneHelper()
    frame_num = helper.frame_num

    gui = ti.GUI('Model Visualizer', res=1024)
    helper.set_transform()
    cur_test_frame_id = 0
    cur_train_frame_id = 0
    similar_particle_num, total_particle_num = helper.set_mesh(cur_test_frame_id, cur_train_frame_id)

    # It maybe too sensitive to press a key. The inflexibility of event control lets me have no idea to adjust it
    # in the circumstance of that level. So, I just put a counter here to make key_press less sensitive.
    key_press_lag = 2
    press_n_counter = 0
    press_m_counter = 0
    press_k_counter = 0
    press_l_counter = 0
    press_s_counter = 0
    while True:
        helper.scene.input(gui)
        gui.get_event()

        if gui.is_pressed('n'):
            if press_n_counter == key_press_lag and cur_test_frame_id < frame_num-1:
                cur_test_frame_id += 1
                press_n_counter = 0
                similar_particle_num, total_particle_num = helper.set_mesh(cur_test_frame_id, cur_train_frame_id)
            else:
                press_n_counter += 1
        elif gui.is_pressed('m'):
            if press_m_counter == key_press_lag and cur_test_frame_id > 0:
                cur_test_frame_id -= 1
                press_m_counter = 0
                similar_particle_num, total_particle_num = helper.set_mesh(cur_test_frame_id, cur_train_frame_id)
            else:
                press_m_counter += 1
        elif gui.is_pressed('r'):
            cur_frame_id = 0
            similar_particle_num, total_particle_num = helper.set_mesh(cur_test_frame_id, cur_train_frame_id)
        elif gui.is_pressed('s'):
            cur_frame_id = int(input("Please input the frame id that you want:"))
            if cur_frame_id < 0 or cur_frame_id > frame_num:
                raise Exception("Input frame id is out of range!")
            similar_particle_num, total_particle_num = helper.set_mesh(cur_test_frame_id, cur_train_frame_id)
        elif gui.is_pressed('c'):
            select_par_id = int(input("Please input the particle id that you want:"))
            if select_par_id < 0 or select_par_id > helper.get_particle_num() - 1:
                raise Exception("Input particle id is out of range!")
            helper.select_particle(select_par_id)
            similar_particle_num, total_particle_num = helper.set_mesh(cur_test_frame_id, cur_train_frame_id)
        elif gui.is_pressed('k'):
            if press_k_counter == key_press_lag and cur_train_frame_id < frame_num-1:
                press_k_counter = 0
                cur_train_frame_id += 1
                similar_particle_num, total_particle_num = helper.set_mesh(cur_test_frame_id, cur_train_frame_id)
            else:
                press_k_counter += 1
        elif gui.is_pressed('l'):
            if press_l_counter == key_press_lag and cur_train_frame_id > 0:
                press_l_counter = 0
                cur_train_frame_id -= 1
                similar_particle_num, total_particle_num = helper.set_mesh(cur_test_frame_id, cur_train_frame_id)
            else:
                press_l_counter += 1

        helper.scene.render()
        gui.set_image(helper.scene.img)
        gui.text("Not rendering", (0.01, 0.99))
        gui.text(f"Current Test Frame:{cur_test_frame_id + helper.start_frame()}", (0.01, 0.95))
        gui.text(f"Current Train Frame:{cur_train_frame_id + helper.start_frame()}", (0.01, 0.91))
        gui.text(f"Similar Particles num:{similar_particle_num}", (0.01, 0.87))
        gui.text(f"Total Particles num:{total_particle_num}", (0.01, 0.83))
        gui.show()
