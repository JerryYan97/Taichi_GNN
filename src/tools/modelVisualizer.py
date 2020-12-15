import taichi as ti
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.reader import read
from Utils.utils_visualization import draw_pd_pn_image

ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)

mesh, dirichlet, mesh_scale, mesh_offset = read(4)
n_particles = mesh.num_vertices
n_elements = mesh.num_faces


if __name__ == "__main__":
    if not os.path.exists("modelVisualization"):
        os.makedirs("modelVisualization")
    for root, dirs, files in os.walk("modelVisualization/"):
        for name in files:
            os.remove(os.path.join(root, name))

    gui = ti.GUI("Model Visualizer", (1024, 1024), background_color=0xf7f7f7)

    while True:
        draw_pd_pn_image(gui, "modelVisualization/mesh.png",
                         mesh.vertices.astype(np.float64)[:, 0:2],
                         mesh.vertices.astype(np.float64)[:, 0:2]
                         , mesh_offset, mesh_scale, mesh.faces, n_elements)



