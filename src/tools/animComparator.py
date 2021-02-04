import taichi as ti
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
# from Simulators.PN import PNSimulation
# from Simulators.PD import PDSimulation
from Utils.reader import read, read_an_obj
from Utils.utils_visualization import rotate_matrix_y_axis, update_boundary_mesh_np


# NOTE: It only works for 3D now
if __name__ == '__main__':
    os.makedirs('results/', exist_ok=True)
    for root, dirs, files in os.walk("results/"):
        for name in files:
            os.remove(os.path.join(root, name))

    # Read in the animation data from SimData/FinalRes/
    PDNN_files_list, PD_files_list, PN_files_list = [], [], []
    # PD-NN
    for _, _, files in os.walk("../../SimData/FinalRes/GNNPDAnimSeq"):
        PDNN_files_list.extend(files)
    PDNN_files_list.sort()
    print("PD-NN Anim files:\n", PDNN_files_list)
    # PD
    for _, _, files in os.walk("../../SimData/FinalRes/ReconstructPDAnimSeq"):
        PD_files_list.extend(files)
    PD_files_list.sort()
    print("PD Anim files:\n", PD_files_list)
    # PN
    for _, _, files in os.walk("../../SimData/FinalRes/ReconstructPNAnimSeq"):
        PN_files_list.extend(files)
    PN_files_list.sort()
    print("PN Anim files:\n", PN_files_list)

    # Input and load the test case
    test_case_id = int(input("Please input the test case ID:"))
    case_info = read(test_case_id)
    frame_num = len(PDNN_files_list)

    # Choose whether to use the video manager
    use_video_manager = int(input("Please choose whether to use the video manager[0--not use/1--use]"))

    # Init scene variables and adjust visualization parameters
    import tina

    # Please comment out the settings of other test cases and write down a note to tell which test case that your
    # transformation belongs to.
    # Test case 1007:
    PD_transform = tina.translate([-1.0, -1.0, -1.0]) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)
    PDGNN_transform = tina.translate([0.0, -1.0, -1.0]) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)
    PN_transform = tina.translate([1.0, -1.0, -1.0]) @ rotate_matrix_y_axis(0.0) @ tina.scale(1.0)

    PD_mesh_pos = ti.Vector.field(3, ti.f32, case_info['mesh'].num_vertices)
    PDGNN_mesh_pos = ti.Vector.field(3, ti.f32, case_info['mesh'].num_vertices)
    PN_mesh_pos = ti.Vector.field(3, ti.f32, case_info['mesh'].num_vertices)

    scene = tina.Scene(culling=False, clipping=True, res=1024)

    PD_mesh = tina.SimpleMesh()
    PD_model = tina.MeshTransform(PD_mesh)

    PDGNN_mesh = tina.SimpleMesh()
    PDGNN_model = tina.MeshTransform(PDGNN_mesh)

    PN_mesh = tina.SimpleMesh()
    PN_model = tina.MeshTransform(PN_mesh)

    scene.add_object(PDGNN_model)
    scene.add_object(PD_model)
    scene.add_object(PN_model)

    if use_video_manager == 1:
        video_manager = ti.VideoManager(output_dir='results/', framerate=12, automatic_build=False)
    gui = ti.GUI('Model Visualizer', res=1024)

    PDGNN_model.set_transform(PDGNN_transform)
    PD_model.set_transform(PD_transform)
    PN_model.set_transform(PN_transform)

    PD_boundary_pos = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)
    PDGNN_boundary_pos = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)
    PN_boundary_pos = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)

    for frame_id in range(frame_num):
        # Read files' pos
        PD_new_pos = np.asarray(read_an_obj("../../SimData/FinalRes/ReconstructPDAnimSeq/" + PD_files_list[frame_id]))
        PDGNN_new_pos = np.asarray(read_an_obj("../../SimData/FinalRes/GNNPDAnimSeq/" + PDNN_files_list[frame_id]))
        PN_new_pos = np.asarray(read_an_obj("../../SimData/FinalRes/ReconstructPNAnimSeq/" + PN_files_list[frame_id]))

        # Update files' pos
        update_boundary_mesh_np(PD_new_pos, PD_boundary_pos, case_info)
        update_boundary_mesh_np(PDGNN_new_pos, PDGNN_boundary_pos, case_info)
        update_boundary_mesh_np(PN_new_pos, PN_boundary_pos, case_info)
        scene.input(gui)
        PD_mesh.set_face_verts(PD_boundary_pos)
        PDGNN_mesh.set_face_verts(PDGNN_boundary_pos)
        PN_mesh.set_face_verts(PN_boundary_pos)
        scene.render()
        gui.set_image(scene.img)
        if use_video_manager == 1:
            video_manager.write_frame(gui.get_image())
            gui.show()
        else:
            file_name_path = "results/" + "anim_comp_" + str(frame_id).zfill(6) + ".png"
            gui.show(file_name_path)
    if use_video_manager == 1:
        video_manager.make_video(gif=True, mp4=True)
