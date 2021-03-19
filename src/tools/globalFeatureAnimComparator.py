import numpy as np
import sys
import os
import pymesh
import tina
import taichi as ti
ti.init(ti.gpu)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.utils_visualization import rotate_matrix_y_axis, update_boundary_pos_np, output_3d_seq
from Utils.reader import read

# Description:
# Used to compare the global feature vector and pd/pn animation.
#
# Instructions:
# n: next frame; m: previous frame;
# SPACE: Rendering from 0 frame to the last frame;
# r: reset to 0 frame;
# s: jump to targe frame. You need to input the frame id in console;
# SceneHelper: you can adjust mesh transformation and mesh input src in this class.


def read_mesh(file_path_name):
    mesh = pymesh.load_mesh(file_path_name)
    boundary_pos = np.ndarray(shape=(mesh.num_faces, 3, 3), dtype=np.float)
    update_boundary_pos_np(mesh.vertices, boundary_pos, mesh.faces, mesh.num_faces)
    return boundary_pos


# NOTE: PD-NN visualization is not tested
class SceneHelper():
    def __init__(self):
        # self.PDNN_file_path = "../../SimData/FinalRes/GNNPDAnimSeq"
        # self.PD_file_path = "../../SimData/FinalRes/ReconstructPDAnimSeq"
        # self.PN_file_path = "../../SimData/FinalRes/ReconstructPNAnimSeq"

        self.PD_file_path = "GlobalFeaVecAnimSeq/PDAnimSeq/"
        self.GFeaVec_file_path = "GlobalFeaVecAnimSeq/GlobalFeaVecAnimSeq/"
        self.PN_file_path = "GlobalFeaVecAnimSeq/PNAnimSeq/"
        # Init files
        self._GFeaVec_files_list, self._PD_files_list, self._PN_files_list = [], [], []
        # PD-NN
        for _, _, files in os.walk(self.GFeaVec_file_path):
            self._GFeaVec_files_list.extend(files)
        self._GFeaVec_files_list.sort()
        print("GFeaVec Anim files:\n", self._GFeaVec_files_list)
        # PD
        for _, _, files in os.walk(self.PD_file_path):
            self._PD_files_list.extend(files)
        self._PD_files_list.sort()
        print("PD Anim files:\n", self._PD_files_list)
        # PN
        for _, _, files in os.walk(self.PN_file_path):
            self._PN_files_list.extend(files)
        self._PN_files_list.sort()
        print("PN Anim files:\n", self._PN_files_list)

        self.frame_num = len(self._PD_files_list)

        self.PD_transform = tina.translate([-1.5, -1.0, -1.0]) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)
        self.GFeaVec_transform = tina.translate([0.0, -1.0, -1.0]) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)
        self.PN_transform = tina.translate([1.5, -1.0, -1.0]) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)

        self.PD_mesh = tina.SimpleMesh()
        self.PD_model = tina.MeshTransform(self.PD_mesh)
        self.PD_mat = tina.Lambert(color=[1.0, 0.5, 0.5])

        self.GFeaVec_mesh = tina.SimpleMesh()
        self.GFeaVec_model = tina.MeshTransform(self.GFeaVec_mesh)
        self.GFeaVec_mat = tina.Lambert(color=[0.5, 1.0, 0.5])

        self.PN_mesh = tina.SimpleMesh()
        self.PN_model = tina.MeshTransform(self.PN_mesh)
        self.PN_mat = tina.Lambert(color=[0.5, 0.5, 1.0])

        self.scene = tina.Scene(culling=False, clipping=True, res=1024)

        self.scene.add_object(self.GFeaVec_model, self.GFeaVec_mat)
        self.scene.add_object(self.PD_model, self.PD_mat)
        self.scene.add_object(self.PN_model, self.PN_mat)

    def set_transform(self):
        self.GFeaVec_model.set_transform(self.GFeaVec_transform)
        self.PD_model.set_transform(self.PD_transform)
        self.PN_model.set_transform(self.PN_transform)

    def set_mesh(self, cur_frame_id):
        # Set Mesh
        PD_file_path_name = self.PD_file_path + "/" + self._PD_files_list[cur_frame_id]
        PD_boundary_pos = read_mesh(PD_file_path_name)
        self.PD_mesh.set_face_verts(PD_boundary_pos)

        GFeaVec_file_path_name = self.GFeaVec_file_path + "/" + self._GFeaVec_files_list[cur_frame_id]
        GFeaVec_boundary_pos = read_mesh(GFeaVec_file_path_name)
        self.GFeaVec_mesh.set_face_verts(GFeaVec_boundary_pos)

        PN_file_path_name = self.PN_file_path + "/" + self._PN_files_list[cur_frame_id]
        PN_boundary_pos = read_mesh(PN_file_path_name)
        self.PN_mesh.set_face_verts(PN_boundary_pos)

    def start_frame(self):
        file_name_len = len(self._PD_files_list[0])
        file_id_str = self._PD_files_list[0][file_name_len - 9:file_name_len - 4]
        file_id = int(file_id_str)
        return file_id


def output_3d_results(pn_x, gfeat_x, pd_x, f, case_info):
    name_pd = "GlobalFeaVecAnimSeq/PDAnimSeq/PD_" + case_info['case_name'] + "_" + str(f).zfill(6) + ".obj"
    name_gfeat = "GlobalFeaVecAnimSeq/GlobalFeaVecAnimSeq/GFeat_" + case_info['case_name'] + "_" + str(f).zfill(6) + ".obj"
    name_pn = "GlobalFeaVecAnimSeq/PNAnimSeq/PN_" + case_info['case_name'] + "_" + str(f).zfill(6) + ".obj"
    output_3d_seq(pd_x, case_info['boundary'][2], name_pd)
    output_3d_seq(gfeat_x, case_info['boundary'][2], name_gfeat)
    output_3d_seq(pn_x, case_info['boundary'][2], name_pn)


# NOTE: It only works for 3D now
if __name__ == '__main__':
    case_info = read(1009)
    b_pts_idx = np.sort(np.fromiter(case_info['boundary'][0], int))

    os.makedirs('results/', exist_ok=True)
    for root, dirs, files in os.walk("results/"):
        for name in files:
            os.remove(os.path.join(root, name))

    os.makedirs('GlobalFeaVecAnimSeq/', exist_ok=True)
    for root, dirs, files in os.walk("GlobalFeaVecAnimSeq/"):
        for name in files:
            os.remove(os.path.join(root, name))
    os.makedirs('GlobalFeaVecAnimSeq/PDAnimSeq/', exist_ok=True)
    os.makedirs('GlobalFeaVecAnimSeq/PNAnimSeq/', exist_ok=True)
    os.makedirs('GlobalFeaVecAnimSeq/GlobalFeaVecAnimSeq/', exist_ok=True)

    # Use Testing data and PreGen Global Feature vector to generate animation
    test_file_path = "../../SimData/TrainingData"
    gfeat_file_path = "../../SimData/PreGenSpannedGlobalFeatureVec"

    test_files_name = []
    for _, _, files in os.walk(test_file_path):
        test_files_name.extend(files)
    test_files_name.sort()
    print("Read test files: ", test_files_name)

    gfeat_files_name = []
    for _, _, files in os.walk(gfeat_file_path):
        gfeat_files_name.extend(files)
    gfeat_files_name.sort()
    print("Read gfeat files: ", gfeat_files_name)

    pn_pos = case_info["mesh"].vertices[b_pts_idx, 0:3]
    gfeat_pos = case_info["mesh"].vertices[b_pts_idx, 0:3]
    pd_pos = case_info["mesh"].vertices[b_pts_idx, 0:3]
    for f in range(len(test_files_name)):
        test_file = np.genfromtxt(test_file_path + "/" + test_files_name[f], delimiter=',')
        gfeat_file = np.genfromtxt(gfeat_file_path + "/" + gfeat_files_name[f], delimiter=',')
        pd_pos_delta = test_file[:, 0:3]
        pn_pos_delta = test_file[:, 3:6]
        gfeat_pos_delta = gfeat_file + pd_pos_delta
        pn_pos = pd_pos + pn_pos_delta[b_pts_idx, 0:3]
        gfeat_pos = pd_pos + gfeat_pos_delta[b_pts_idx, 0:3]
        pd_pos += pd_pos_delta[b_pts_idx, 0:3]

        whole_gfeat_pos = np.zeros((case_info["mesh"].num_vertices, 3), dtype=float)
        whole_pd_pos = np.zeros((case_info["mesh"].num_vertices, 3), dtype=float)
        whole_pn_pos = np.zeros((case_info["mesh"].num_vertices, 3), dtype=float)
        whole_gfeat_pos[b_pts_idx, 0:3] = gfeat_pos
        whole_pd_pos[b_pts_idx, 0:3] = pd_pos
        whole_pn_pos[b_pts_idx, 0:3] = pn_pos

        output_3d_results(whole_pn_pos, whole_gfeat_pos, whole_pd_pos, f + 1, case_info)
        print("Generated frame ", f + 1)

    print("Finish Anim seq generation.")

    # Init scene variables and adjust visualization parameters
    helper = SceneHelper()
    frame_num = helper.frame_num

    gui = ti.GUI('Model Visualizer', res=1024)
    helper.set_transform()
    cur_frame_id = 0

    helper.set_mesh(cur_frame_id)

    # It maybe too sensitive to press a key. The inflexibility of event control lets me have no idea to adjust it
    # in the circumstance of that level. So, I just put a counter here to make key_press less sensitive.
    key_press_lag = 2
    press_n_counter = 0
    press_m_counter = 0
    press_s_counter = 0
    while True:
        helper.scene.input(gui)
        gui.get_event()

        if gui.is_pressed('n'):
            if press_n_counter == key_press_lag and cur_frame_id < frame_num-1:
                cur_frame_id += 1
                press_n_counter = 0
                helper.set_mesh(cur_frame_id)
            else:
                press_n_counter += 1
        elif gui.is_pressed('m'):
            if press_m_counter == key_press_lag and cur_frame_id > 0:
                cur_frame_id -= 1
                press_m_counter = 0
                helper.set_mesh(cur_frame_id)
            else:
                press_m_counter += 1
        elif gui.is_pressed('r'):
            cur_frame_id = 0
            helper.set_mesh(cur_frame_id)
        elif gui.is_pressed('s'):
            cur_frame_id = int(input("Please input the frame id that you want:"))
            if cur_frame_id < 0 or cur_frame_id > frame_num:
                raise Exception("Input frame id is out of range!")
            helper.set_mesh(cur_frame_id)
        elif gui.is_pressed(ti.GUI.SPACE):
            # Rendering to GIF/MP4 from start
            video_manager = ti.VideoManager(output_dir='results/', framerate=24)
            for cur_frame_id in range(frame_num):
                helper.set_mesh(cur_frame_id)

                helper.scene.render()
                gui.set_image(helper.scene.img)
                video_manager.write_frame(gui.get_image())
                gui.text("Rendering...", (0.01, 0.99))
                gui.text(f"Current Frame:{cur_frame_id + helper.start_frame()}. Total Frame:{frame_num - 1}",
                         (0.01, 0.95))
                gui.show(f'results/frame_{cur_frame_id}.png')
            video_manager.make_video(gif=True, mp4=True)
            break

        helper.scene.render()
        gui.set_image(helper.scene.img)
        gui.text("Not rendering", (0.01, 0.99))
        gui.text(f"Current Frame:{cur_frame_id + helper.start_frame()}", (0.01, 0.95))
        gui.show()
