# import taichi_three as t3
import taichi as ti
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.reader import read
from Utils.utils_visualization import draw_pd_pn_image, set_3D_scene, update_boundary_pos


if __name__ == "__main__":
    if not os.path.exists("modelVisualization"):
        os.makedirs("modelVisualization")
    for root, dirs, files in os.walk("modelVisualization/"):
        for name in files:
            os.remove(os.path.join(root, name))

    case_info = read(1001)
    n_particles = case_info['mesh'].num_vertices
    n_elements = 0
    if case_info['dim'] == 2:
        n_elements = case_info['mesh'].num_faces
    elif case_info['dim'] == 3:
        n_elements = case_info['mesh'].num_elements

    gui = None
    camera = None
    model = None
    scene = None

    if case_info['dim'] == 2:
        gui = ti.GUI("Model Visualizer", (1024, 1024), background_color=0xf7f7f7)
    elif case_info['dim'] == 3:
        import tina
        boundary_points, boundary_edges, boundary_triangles = case_info['boundary']
        mesh_pos = ti.Vector.field(3, ti.f32, case_info['mesh'].num_vertices)
        scene = tina.Scene(culling=False)
        mesh = tina.SimpleMesh()
        scene.add_object(mesh)
        gui = ti.GUI('Model Visualizer')

        mesh_pos.from_numpy(case_info['mesh'].vertices)
        boundary_pos = np.ndarray(shape=(len(boundary_triangles), 3, 3), dtype=np.float)
        boundary_tri_num = len(boundary_triangles)

    while True:
        if case_info['dim'] == 2:
            draw_pd_pn_image(gui, "modelVisualization/mesh.png",
                            case_info['mesh'].vertices.astype(np.float64)[:, 0:2],
                            case_info['mesh'].vertices.astype(np.float64)[:, 0:2],
                            case_info['mesh_offset'], case_info['mesh_scale'], case_info['mesh'].faces, n_elements)
        elif case_info['dim'] == 3:
            update_boundary_pos(mesh_pos, boundary_pos, boundary_triangles, boundary_tri_num)
            scene.input(gui)
            mesh.set_face_verts(boundary_pos)
            scene.render()
            gui.set_image(scene.img)
            gui.show()
