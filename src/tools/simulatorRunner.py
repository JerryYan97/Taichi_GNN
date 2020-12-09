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

test_case = 1
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

    video_manager = ti.VideoManager(output_dir='PD_PN_Compare/', framerate=24, automatic_build=False)

    pd_simulator = PDSimulation(test_case, 2)
    pn_simulator = PNSimulation(int(test_case), 2)
    pd_simulator.set_force(-45, 3)
    pn_simulator.set_force(-45, 3)
    pd_simulator.set_material(rho, E, nu, dt)
    pn_simulator.set_material(rho, E, nu, dt)
    pd_simulator.compute_restT_and_m()
    pn_simulator.compute_restT_and_m()
    pn_simulator.zero.fill(0)

    gui = ti.GUI("Model Visualizer", (1024, 1024), background_color=0xf7f7f7)
    filename = f'./modelVisualization/frame_{0:05d}.png'
    draw_pd_pn_image(gui, filename,
                     pd_simulator.pos.to_numpy(), pn_simulator.x.to_numpy(),
                     mesh_offset, mesh_scale,
                     mesh.faces, n_elements,
                     True, video_manager)

    for i in range(frame_count):
        print("//////////////////////////////////////Frame ", i, "/////////////////////////////////")
        _, pd_pos_taichi, pd_vel_taichi = pd_simulator.data_one_frame(pd_simulator.pos, pd_simulator.vel)
        _, pn_pos_taichi, pn_vel_taichi = pn_simulator.data_one_frame(pn_simulator.x, pn_simulator.v)
        filename = f'./modelVisualization/frame_{i + 1:05d}.png'
        draw_pd_pn_image(gui, filename,
                         pd_simulator.pos.to_numpy(), pn_simulator.x.to_numpy(),
                         mesh_offset, mesh_scale,
                         mesh.faces, n_elements,
                         True, video_manager)

video_manager.make_video(gif=True, mp4=True)
