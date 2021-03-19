import sys, os, time
import taichi as ti
import numpy as np
import pymesh
from src.Utils.reader import *
from src.Utils.utils_visualization import output_3d_seq, update_boundary_mesh_np


# PN result: Red (Top Layer)
# Corrected PD result: Green (Second Layer)
# PD result: Blue (Bottom Layer)
def write_combined_image(pn_x, corrected_pd_x, pd_x):
    particle_pos_pn = (pn_x + mesh_offset) * mesh_scale
    particle_pos_cor_pd = (corrected_pd_x + mesh_offset) * mesh_scale
    particle_pos_pd = (pd_x + mesh_offset) * mesh_scale

    for i in range(n_elements):
        for j in range(3):
            a, b = mesh.faces[i, j], mesh.faces[i, (j + 1) % 3]
            # PD
            # gui.line((particle_pos_pd[a][0], particle_pos_pd[a][1]),
            #          (particle_pos_pd[b][0], particle_pos_pd[b][1]),
            #          radius=1,
            #          color=0xFF0000)
            # Corrected PD
            gui.line((particle_pos_cor_pd[a][0], particle_pos_cor_pd[a][1]),
                     (particle_pos_cor_pd[b][0], particle_pos_cor_pd[b][1]),
                     radius=1,
                     color=0x00FF00)
            # PN
            gui.line((particle_pos_pn[a][0], particle_pos_pn[a][1]),
                     (particle_pos_pn[b][0], particle_pos_pn[b][1]),
                     radius=1,
                     color=0x0000FF)
    video_manager.write_frame(gui.get_image())
    gui.show()


def output_3d_results(pn_x, corrected_pd_x, pd_x, f, case_info):
    name_pd = "SimData/FinalRes/ReconstructPDAnimSeq/PD_" + case_info['case_name'] + "_" + str(f).zfill(6) + ".obj"
    name_pd_gnn = "SimData/FinalRes/GNNPDAnimSeq/PDGNN_" + case_info['case_name'] + "_" + str(f).zfill(6) + ".obj"
    name_pn = "SimData/FinalRes/ReconstructPNAnimSeq/PN_" + case_info['case_name'] + "_" + str(f).zfill(6) + ".obj"
    output_3d_seq(pd_x, case_info['boundary'][2], name_pd)
    output_3d_seq(corrected_pd_x, case_info['boundary'][2], name_pd_gnn)
    output_3d_seq(pn_x, case_info['boundary'][2], name_pn)


if __name__ == "__main__":
    case_info = read(1009)
    mesh = case_info['mesh']
    dirichlet = case_info['dirichlet']
    mesh_scale = case_info['mesh_scale']
    mesh_offset = case_info['mesh_offset']
    dim = case_info['dim']
    n_particles = mesh.num_vertices
    if dim == 2:
        n_elements = mesh.num_faces
    else:
        n_elements = mesh.num_elements

    ti.init(arch=ti.gpu, default_fp=ti.f64)

    directory = os.getcwd() + '/SimData/TmpRenderedImgs/' + '_'.join(os.path.basename(sys.argv[0]) + "_combined") + '/'
    print('Tmp rendered results directory:', directory)
    testpath = "SimData/RunNNRes"

    # Delete old results in the SimData/FinalRes folder
    for root, dirs, files in os.walk("SimData/FinalRes/GNNPDAnimSeq"):
        for name in files:
            os.remove(os.path.join(root, name))
    for root, dirs, files in os.walk("SimData/FinalRes/ReconstructPDAnimSeq"):
        for name in files:
            os.remove(os.path.join(root, name))
    for root, dirs, files in os.walk("SimData/FinalRes/ReconstructPNAnimSeq"):
        for name in files:
            os.remove(os.path.join(root, name))

    # Init 2d rendering settings
    if dim == 2:
        video_manager = ti.VideoManager(output_dir=directory + 'images/', framerate=24, automatic_build=False)
        gui = ti.GUI("Test Visualizer", (1024, 1024), background_color=0xf7f7f7)
        write_combined_image(mesh.vertices, mesh.vertices, mesh.vertices)
    else:
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

    # Read files
    TestResults_Files = []
    for _, _, files in os.walk(testpath):
        TestResults_Files.extend(files)
    b_pts_idx_name = "b_pts_idx_" + case_info['case_name'] + ".csv"
    b_pts_file_idx = TestResults_Files.index(b_pts_idx_name)
    del TestResults_Files[b_pts_file_idx]
    TestResults_Files.sort()
    print("TestResults_Files:", TestResults_Files)
    b_pts_idx = np.genfromtxt(testpath + "/" + b_pts_idx_name, delimiter=',', dtype=int)

    # Reconstruct pos:
    print("You are now selecting the test case " + case_info['case_name'])
    frame_id = int(input("Please input the frame id that you want to start:"))
    start_mesh_path = "./SimData/StartFrame/" + case_info['case_name'] + "_frame_" + f'{frame_id:05}' + ".obj"
    if not os.path.isfile(start_mesh_path):
        raise Exception("The start mesh path doesn't exist!")
    start_mesh = pymesh.load_mesh(start_mesh_path)

    # pn_pos = mesh.vertices[b_pts_idx, 0:dim]
    # corrected_pd_pos = mesh.vertices[b_pts_idx, 0:dim]
    # pd_pos = mesh.vertices[b_pts_idx, 0:dim]
    pn_pos = start_mesh.vertices[b_pts_idx, 0:dim]
    corrected_pd_pos = start_mesh.vertices[b_pts_idx, 0:dim]
    pd_pos = start_mesh.vertices[b_pts_idx, 0:dim]
    for f in range(len(TestResults_Files)):
        test_file = np.genfromtxt(testpath + "/" + TestResults_Files[f], delimiter=',')

        pd_pos_delta = test_file[:, 0:dim]
        corrected_pd_pos_delta = test_file[:, dim:dim * 2] + pd_pos_delta
        pn_pos_delta = test_file[:, dim * 2:dim * 3] + pd_pos_delta

        # PN -> PD: pd_pos[n+1] is calculated by pn_pos[n] + pd_dis.
        # corrected_pd_pos = pn_pos + corrected_pd_pos_delta
        # pd_pos = pn_pos + pd_pos_delta
        # pn_pos = pn_pos + pn_pos_delta

        # PD -> PN: pn_pos[n+1] is calculated by pd_pos[n] + pn_dis.
        corrected_pd_pos = corrected_pd_pos_delta + pd_pos
        pn_pos = pn_pos_delta + pd_pos
        pd_pos = pd_pos_delta + pd_pos

        whole_corrected_pd_pos = np.zeros((mesh.num_vertices, dim), dtype=float)
        whole_pd_pos = np.zeros((mesh.num_vertices, dim), dtype=float)
        whole_pn_pos = np.zeros((mesh.num_vertices, dim), dtype=float)
        whole_corrected_pd_pos[b_pts_idx, 0:dim] = corrected_pd_pos
        whole_pd_pos[b_pts_idx, 0:dim] = pd_pos
        whole_pn_pos[b_pts_idx, 0:dim] = pn_pos

        if dim == 2:
            write_combined_image(whole_pn_pos, corrected_pd_pos, pd_pos)
        else:
            output_3d_results(whole_pn_pos, whole_corrected_pd_pos, whole_pd_pos, f + 1 + frame_id, case_info)
            update_boundary_mesh_np(whole_corrected_pd_pos, scene_info['boundary_pos'], case_info)
            scene_info['scene'].input(gui)
            scene_info['tina_mesh'].set_face_verts(scene_info['boundary_pos'])
            scene_info['scene'].render()
            gui.set_image(scene_info['scene'].img)
            gui.show()

    if dim == 2:
        video_manager.make_video(gif=True, mp4=True)
