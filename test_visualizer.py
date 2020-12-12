import sys, os, time
import taichi as ti
import numpy as np
import pymesh
from src.Utils.reader import *

##############################################################################

mesh, dirichlet, mesh_scale, mesh_offset = read(1)

directory = os.getcwd() + '/output/' + '_'.join(os.path.basename(sys.argv[0]) + "_combined") + '/'

print('output directory:', directory)
video_manager = ti.VideoManager(output_dir=directory + 'images/', framerate=24, automatic_build=False)

##############################################################################

ti.init(arch=ti.gpu, default_fp=ti.f64)

n_particles = mesh.num_vertices
n_elements = mesh.num_faces


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
            gui.line((particle_pos_pd[a][0], particle_pos_pd[a][1]),
                     (particle_pos_pd[b][0], particle_pos_pd[b][1]),
                     radius=1,
                     color=0xFF0000)
            # Corrected PD
            gui.line((particle_pos_cor_pd[a][0], particle_pos_cor_pd[a][1]),
                     (particle_pos_cor_pd[b][0], particle_pos_cor_pd[b][1]),
                     radius=1,
                     color=0x00FF00)
            # PN
            # gui.line((particle_pos_pn[a][0], particle_pos_pn[a][1]),
            #          (particle_pos_pn[b][0], particle_pos_pn[b][1]),
            #          radius=1,
            #          color=0x0000FF)
    video_manager.write_frame(gui.get_image())
    gui.show()


if __name__ == "__main__":
    testpath = "TestResult"
    realpath = "Outputs_T"

    gui = ti.GUI("Test Visualizer", (1024, 1024), background_color=0xf7f7f7)

    write_combined_image(mesh.vertices, mesh.vertices, mesh.vertices)

    # TODO: Wait for PN debug:
    TestResults_Files = []
    for _, _, files in os.walk(testpath):
        TestResults_Files.extend(files)
    TestResults_Files.sort()
    print("TestResults_Files:", TestResults_Files)

    # Init pos:
    pn_pos, corrected_pd_pos, pd_pos = mesh.vertices[:, 0:2], mesh.vertices[:, 0:2], mesh.vertices[:, 0:2]
    for f in range(len(TestResults_Files)):
        # test_file = np.genfromtxt("{}{}".format(testpath + "/", TestResults_Files[f]), delimiter=' ')
        test_file = np.genfromtxt(testpath + "/" + TestResults_Files[f], delimiter=',')

        # test_file = np.genfromtxt("{}{}".format(realpath + "/", TestResults_Files[f]), delimiter=' ')
        pd_pos_delta = test_file[:, 0:2]
        corrected_pd_pos_delta = test_file[:, 2:4] + pd_pos_delta
        pn_pos_delta = test_file[:, 6:8] + pd_pos_delta
        # pn_pos_delta = test_file[:, 2:4]

        # PN -> PD: pd_pos[n+1] is calculated by pn_pos[n] + pd_dis.
        # corrected_pd_pos = pn_pos + corrected_pd_pos_delta
        # pd_pos = pn_pos + pd_pos_delta
        # pn_pos = pn_pos + pn_pos_delta

        # PD -> PN: pn_pos[n+1] is calculated by pd_pos[n] + pn_dis.
        corrected_pd_pos = corrected_pd_pos_delta + pd_pos
        pn_pos = pn_pos_delta + pd_pos
        pd_pos = pd_pos_delta + pd_pos

        write_combined_image(pn_pos, corrected_pd_pos, pd_pos)

video_manager.make_video(gif=True, mp4=True)