import sys, os, time
import taichi as ti
import numpy as np
import pymesh
from src.Utils.reader import *

##############################################################################

mesh, dirichlet, mesh_scale, mesh_offset = read(1)

directory = ""
visualMode = 3
if visualMode is 1:
    directory = os.getcwd() + '/output/' + '_'.join(os.path.basename(sys.argv[0])) + '/'
elif visualMode is 0:
    directory = os.getcwd() + '/output/' + '_'.join(os.path.basename(sys.argv[0])+"_ref") + '/'
elif visualMode is 2:
    directory = os.getcwd() + '/output/' + '_'.join(os.path.basename(sys.argv[0])+"_pd") + '/'
elif visualMode is 3:
    directory = os.getcwd() + '/output/' + '_'.join(os.path.basename(sys.argv[0]) + "_combined") + '/'
os.makedirs(directory + 'images/', exist_ok=True)
print('output directory:', directory)
video_manager = ti.VideoManager(output_dir=directory + 'images/', framerate=24, automatic_build=False)

##############################################################################

ti.init(arch=ti.gpu, default_fp=ti.f64)

real = ti.f64
dim = 2

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)

n_particles = mesh.num_vertices
n_elements = mesh.num_faces

x = vec()

deltap = ti.Vector.field(dim, real, n_particles)

vertices = ti.var(ti.i32)
ti.root.dense(ti.k, n_particles).place(x)

ti.root.dense(ti.ij, (n_elements, 3)).place(vertices)


def write_image(f):
    if visualMode is 0:
        col = 0x4FB99F
    if visualMode is 1:
        col = 0xE74C3C
    if visualMode is 2:
        col = 0xF4D03F
    particle_pos = (x.to_numpy() + mesh_offset) * mesh_scale
    for i in range(n_elements):
        for j in range(3):
            a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
            gui.line((particle_pos[a][0], particle_pos[a][1]),
                     (particle_pos[b][0], particle_pos[b][1]),
                     radius=1,
                     color=col)
    for i in dirichlet:
        gui.circle(particle_pos[i], radius=3, color=0x44FFFF)
    # video_manager.write_frame(gui.get_image())
    gui.show(directory + f'images/{f:06d}.png')


# PN result: Red (Top Layer)
# Corrected PD result: Green (Second Layer)
# PD result: Blue (Bottom Layer)
def write_combined_image(f, pn_x, corrected_pd_x, pd_x):
    particle_pos_pn = (pn_x + mesh_offset) * mesh_scale
    particle_pos_cor_pd = (corrected_pd_x + mesh_offset) * mesh_scale
    particle_pos_pd = (pd_x + mesh_offset) * mesh_scale

    for i in range(n_elements):
        for j in range(3):
            a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
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
    gui.show(directory + f'images/{f:06d}.png')


if __name__ == "__main__":
    testpath = "TestResult"
    realpath = "Outputs_T"

    x.from_numpy(mesh.vertices.astype(np.float64))
    vertices.from_numpy(mesh.faces)
    gui = ti.GUI("MPM_TEST", (1024, 1024), background_color=0x112F41)
    vertices_ = vertices.to_numpy()

    if visualMode != 3:
        write_image(0)
    else:
        write_combined_image(0, mesh.vertices, mesh.vertices, mesh.vertices)

    Files_Global = []
    Files_Global2 = []
    if visualMode == 1:
        for _, _, files in os.walk(testpath):
            Files_Global.extend(files)
        Files_Global.sort()

        for _, _, files in os.walk(realpath):
            Files_Global2.extend(files)
        Files_Global2.sort()

    if visualMode is 0 or visualMode is 2:
        for _, _, files in os.walk(realpath):
            Files_Global2.extend(files)
        Files_Global2.sort()

    # TODO: Wait for PN debug:
    TestResults_Files = []
    if visualMode is 3:
        for _, _, files in os.walk(testpath):
            TestResults_Files.extend(files)
        TestResults_Files.sort()
        print("TestResults_Files:", TestResults_Files)

    # TestResults_Files = []
    # if visualMode is 3:
    #     for _, _, files in os.walk(realpath):
    #         TestResults_Files.extend(files)
    #     TestResults_Files.sort()
    #     print("TestResults_Files:", TestResults_Files)

    ff = 0
    origin = x.to_numpy().reshape((mesh.num_vertices, 2)).astype(np.float32)

    if visualMode == 1:
        for f in range(len(Files_Global)):  # pn from network
            fperframe = np.genfromtxt("{}{}".format(testpath + "/", Files_Global[f]), dtype=np.dtype(str))
            fperframe2 = np.genfromtxt("{}{}".format(realpath + "/", Files_Global2[f]), dtype=np.dtype(str))
            pdpos = fperframe[:, 0:2].astype(float)  # a[start:stop] items start through stop-1
            deltapos = fperframe[:, 2:4].astype(float)  # a[start:stop] items start through stop-1
            pnposr = fperframe2[:, 2:4].astype(float)  # a[start:stop] items start through stop-1
            pnpos = pdpos + deltapos
            pos = pnpos + origin
            origin = pnposr + origin
            x.from_numpy(pos.astype(np.float32))
            write_image(ff + 1)
            video_manager.write_frame(gui.get_image())
            ff = ff + 1

    if visualMode == 0:
        for f in range(len(Files_Global2)):  # pn reference
            fperframe = np.genfromtxt("{}{}".format(realpath + "/", Files_Global2[f]), dtype=np.dtype(str))
            pnpos = fperframe[:, 2:4].astype(float)  # a[start:stop] items start through stop-1
            pos = pnpos + origin
            origin = pnpos + origin
            x.from_numpy(pos.astype(np.float32))
            write_image(ff + 1)
            video_manager.write_frame(gui.get_image())
            ff = ff + 1

    if visualMode == 2:
        # for f in range(len(Files_Global2)):  # pd reference
        #     fperframe = np.genfromtxt("{}{}".format(realpath + "/", Files_Global2[f]), dtype=np.dtype(str))
        #     pnpos = fperframe[:, 0:2].astype(float)  # a[start:stop] items start through stop-1
        #     pos = pnpos + origin
        #     origin = pnpos + origin
        #     x.from_numpy(pos.astype(np.float32))
        #     write_image(ff + 1)
        #     video_manager.write_frame(gui.get_image())
        #     ff = ff + 1
        for f in range(len(Files_Global2)):  # pd reference
            fperframe = np.genfromtxt("{}{}".format(realpath + "/", Files_Global2[f]), dtype=np.dtype(str))
            pnpos = fperframe[:, 2:4].astype(float)  # a[start:stop] items start through stop-1
            pdpos = fperframe[:, 0:2].astype(float)  # a[start:stop] items start through stop-1
            pos = pdpos + origin
            origin = pnpos + origin
            x.from_numpy(pos.astype(np.float32))
            write_image(ff + 1)
            video_manager.write_frame(gui.get_image())
            ff = ff + 1

    if visualMode == 3:
        # Init pos:
        pn_pos, corrected_pd_pos, pd_pos = mesh.vertices[:, 0:2], mesh.vertices[:, 0:2], mesh.vertices[:, 0:2]
        for f in range(len(TestResults_Files)):
            test_file = np.genfromtxt("{}{}".format(testpath + "/", TestResults_Files[f]), delimiter=' ')
            # test_file = np.genfromtxt("{}{}".format(realpath + "/", TestResults_Files[f]), delimiter=' ')
            pd_pos_delta = test_file[:, 0:2]
            corrected_pd_pos_delta = test_file[:, 2:4]
            pn_pos_delta = test_file[:, 6:8]
            # pn_pos_delta = test_file[:, 2:4]

            # PN -> PD: pd_pos[n+1] is calculated by pn_pos[n] + pd_dis.
            # corrected_pd_pos = pn_pos + corrected_pd_pos_delta
            # pd_pos = pn_pos + pd_pos_delta
            # pn_pos = pn_pos + pn_pos_delta

            # PD -> PN: pn_pos[n+1] is calculated by pd_pos[n] + pn_dis.
            corrected_pd_pos = corrected_pd_pos_delta + pd_pos
            pn_pos = pn_pos_delta + pd_pos
            pd_pos = pd_pos_delta + pd_pos

            write_combined_image(f, pn_pos, corrected_pd_pos, pd_pos)

    video_manager.make_video(gif=True, mp4=True)