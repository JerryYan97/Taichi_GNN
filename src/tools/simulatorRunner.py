import taichi as ti
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Simulators.PN import PNSimulation
from Simulators.PD import PDSimulation
from Utils.reader import read
from Utils.utils_visualization import draw_pd_pn_image

ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

test_case = 2
mesh, dirichlet, mesh_scale, mesh_offset = read(test_case)
n_particles = mesh.num_vertices
n_elements = mesh.num_faces

rho = 1e2
E = 1e4
nu = 0.4
dt = 0.01

frame_count = 50

if __name__ == '__main__':
    if not os.path.exists("PD_PN_Compare"):
        os.makedirs("PD_PN_Compare")
    for root, dirs, files in os.walk("PD_PN_Compare/"):
        for name in files:
            os.remove(os.path.join(root, name))

    # PN debug:
    pn = PNSimulation(int(test_case), 2)
    pn.set_force(-45.0, 1.0)
    pn.Run(1, frame_count)

