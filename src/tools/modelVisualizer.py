import taichi as ti
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.reader import read
from Utils.utils_visualization import draw_pd_pn_image, update_boundary_mesh


if __name__ == "__main__":
    if not os.path.exists("modelVisualization"):
        os.makedirs("modelVisualization")
    for root, dirs, files in os.walk("modelVisualization/"):
        for name in files:
            os.remove(os.path.join(root, name))

    case_info = read(1010)
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
        mesh_pos = ti.Vector.field(3, ti.f32, case_info['mesh'].num_vertices)
        scene = tina.Scene(culling=False, clipping=True)
        mesh = tina.SimpleMesh()
        model = tina.MeshTransform(mesh)
        scene.add_object(model)
        gui = ti.GUI('Model Visualizer')

        # model.set_transform(case_info['transformation_mat'])
        mesh_pos.from_numpy(case_info['mesh'].vertices)
        boundary_pos = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)

    while True:
        if case_info['dim'] == 2:
            draw_pd_pn_image(gui, "modelVisualization/mesh.png",
                             case_info['mesh'].vertices.astype(np.float64)[:, 0:2],
                             case_info['mesh'].vertices.astype(np.float64)[:, 0:2],
                             case_info['mesh_offset'], case_info['mesh_scale'], case_info['mesh'].faces, n_elements)
        elif case_info['dim'] == 3:
            update_boundary_mesh(mesh_pos, boundary_pos, case_info)
            scene.input(gui)
            mesh.set_face_verts(boundary_pos)
            scene.render()
            gui.set_image(scene.img)
            gui.show()
