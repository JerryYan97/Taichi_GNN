import sys, os, time
import taichi as ti
import numpy as np
import pymesh
from src.Utils.reader import *
from src.Utils.utils_visualization import output_3d_seq, update_boundary_mesh_np


def output_3d_results(pd_x, f, case_info):
    name_pd = "../../SimData/FinalRes/TrainReconstructPDAnimSeq/PD_" + case_info['case_name'] + "_" + str(f).zfill(6) + ".obj"
    output_3d_seq(pd_x, case_info['boundary'][2], name_pd)


if __name__ == "__main__":
    case_info = read(1011)
    mesh = case_info['mesh']
    dim = case_info['dim']
    n_particles = mesh.num_vertices
    n_elements = mesh.num_elements

    ti.init(arch=ti.gpu, default_fp=ti.f64)

    # Delete old results in the SimData/FinalRes folder
    os.makedirs('../../SimData/FinalRes/TrainReconstructPDAnimSeq/', exist_ok=True)
    for root, dirs, files in os.walk("../../SimData/FinalRes/TrainReconstructPDAnimSeq"):
        for name in files:
            os.remove(os.path.join(root, name))

    # Tina temporarily doesn't support wireframe/transparent materials. So we will just show the pd_gnn result for the
    # purpose of debugging.
    import tina
    scene_info = {}
    scene_info['scene'] = tina.Scene(culling=False, clipping=True)
    scene_info['tina_mesh'] = tina.SimpleMesh()
    scene_info['model'] = tina.MeshTransform(scene_info['tina_mesh'])
    scene_info['scene'].add_object(scene_info['model'])
    scene_info['boundary_pos'] = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)
    gui = ti.GUI('Test Visualizer')

    # Reconstruct pos:
    print("You are now selecting the test case " + case_info['case_name'])
    frame_id = int(input("Please input the frame id that you want to start:"))
    start_mesh_path = "../../SimData/StartFrame/" + case_info['case_name'] + "_frame_" + f'{frame_id:05}' + ".obj"
    if not os.path.isfile(start_mesh_path):
        raise Exception("The start mesh path doesn't exist!")
    start_mesh = pymesh.load_mesh(start_mesh_path)

    pd_pos = start_mesh.vertices

    # Read files:
    Training_Files_Path = "../../SimData/TrainingData"
    Training_Files = []
    for _, _, files in os.walk(Training_Files_Path):
        Training_Files.extend(files)
    Training_Files.sort()

    for f in range(len(Training_Files)):
        train_file = np.genfromtxt(Training_Files_Path + "/" + Training_Files[f], delimiter=',')
        pd_pos_delta = train_file[:, 0:dim]
        # PD -> PN: pn_pos[n+1] is calculated by pd_pos[n] + pn_dis.
        pd_pos = pd_pos_delta + pd_pos
        output_3d_results(pd_pos, f + 1 + frame_id, case_info)
        update_boundary_mesh_np(pd_pos, scene_info['boundary_pos'], case_info)
        scene_info['scene'].input(gui)
        scene_info['tina_mesh'].set_face_verts(scene_info['boundary_pos'])
        scene_info['scene'].render()
        gui.set_image(scene_info['scene'].img)
        gui.show()

