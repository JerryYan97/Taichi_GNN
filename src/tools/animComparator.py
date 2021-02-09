import numpy as np
import sys
import os
import pymesh
import tina
import taichi as ti
ti.init(ti.gpu)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.utils_visualization import rotate_matrix_y_axis, update_boundary_pos_np


def read_mesh(file_path_name):
    mesh = pymesh.load_mesh(file_path_name)
    boundary_pos = np.ndarray(shape=(mesh.num_faces, 3, 3), dtype=np.float)
    update_boundary_pos_np(mesh.vertices, boundary_pos, mesh.faces, mesh.num_faces)
    return boundary_pos


# NOTE: PD-NN visualization is not tested
class SceneHelper():
    def __init__(self):
        # Init files
        self._PDNN_files_list, self._PD_files_list, self._PN_files_list = [], [], []
        # PD-NN
        # for _, _, files in os.walk("../../SimData/FinalRes/GNNPDAnimSeq"):
        #     self.PDNN_files_list.extend(files)
        # self.PDNN_files_list.sort()
        # print("PD-NN Anim files:\n", self.PDNN_files_list)
        # PD
        for _, _, files in os.walk("../../SimData/PDAnimSeq"):
            self._PD_files_list.extend(files)
        self._PD_files_list.sort()
        print("PD Anim files:\n", self._PD_files_list)
        # PN
        for _, _, files in os.walk("../../SimData/PNAnimSeq"):
            self._PN_files_list.extend(files)
        self._PN_files_list.sort()
        print("PN Anim files:\n", self._PN_files_list)

        self.frame_num = len(self._PD_files_list)

        self.PD_transform = tina.translate([-1.0, -1.0, -1.0]) @ rotate_matrix_y_axis(-90.0) @ tina.scale(1.0)
        # self.PDGNN_transform = tina.translate([0.0, -1.0, -1.0]) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)
        self.PN_transform = tina.translate([1.0, -1.0, -1.0]) @ rotate_matrix_y_axis(-90.0) @ tina.scale(1.0)

        self.PD_mesh = tina.SimpleMesh()
        self.PD_model = tina.MeshTransform(self.PD_mesh)
        self.PD_mat = tina.Lambert(color=[1.0, 0.5, 0.5])

        # self.PDGNN_mesh = tina.SimpleMesh()
        # self.PDGNN_model = tina.MeshTransform(PDGNN_mesh)
        # self.PDGNN_mat = tina.Lambert(color=[0.5, 1.0, 0.5])

        self.PN_mesh = tina.SimpleMesh()
        self.PN_model = tina.MeshTransform(self.PN_mesh)
        self.PN_mat = tina.Lambert(color=[0.5, 0.5, 1.0])

        self.scene = tina.Scene(culling=False, clipping=True, res=1024)

        # scene.add_object(self.PDGNN_model, self.PDGNN_mat)
        self.scene.add_object(self.PD_model, self.PD_mat)
        self.scene.add_object(self.PN_model, self.PN_mat)

    def set_transform(self):
        # self.PDGNN_model.set_transform(self.PDGNN_transform)
        self.PD_model.set_transform(self.PD_transform)
        self.PN_model.set_transform(self.PN_transform)

    def set_mesh(self, cur_frame_id):
        # Set Mesh
        PD_file_path_name = "../../SimData/PDAnimSeq/" + self._PD_files_list[cur_frame_id]
        PD_boundary_pos = read_mesh(PD_file_path_name)
        self.PD_mesh.set_face_verts(PD_boundary_pos)

        # PDGNN_file_path_name = "../../SimData/PDGNNAnimSeq/" + self._PDGNN_files_list[cur_frame_id]
        # PDGNN_boundary_pos = read_mesh(PDGNN_file_path_name)
        # self.PDGNN_mesh.set_face_verts(PDGNN_boundary_pos)

        PN_file_path_name = "../../SimData/PNAnimSeq/" + self._PN_files_list[cur_frame_id]
        PN_boundary_pos = read_mesh(PN_file_path_name)
        self.PN_mesh.set_face_verts(PN_boundary_pos)


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
            if press_n_counter == key_press_lag and cur_frame_id < frame_num:
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
            video_manager = ti.VideoManager(output_dir='results/', framerate=24, automatic_build=False)
            for cur_frame_id in range(frame_num):
                helper.set_mesh(cur_frame_id)

                helper.scene.render()
                gui.set_image(helper.scene.img)
                video_manager.write_frame(gui.get_image())
                gui.text("Rendering...", (0.01, 0.99))
                gui.text(f"Current Frame:{cur_frame_id}. Total Frame:{frame_num - 1}", (0.01, 0.95))
                gui.show()
            video_manager.make_video(gif=True, mp4=True)
            break

        helper.scene.render()
        gui.set_image(helper.scene.img)
        gui.text("Not rendering", (0.01, 0.99))
        gui.text(f"Current Frame:{cur_frame_id}", (0.01, 0.95))
        gui.show()
