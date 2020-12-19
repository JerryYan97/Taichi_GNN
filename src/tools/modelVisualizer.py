import taichi as ti
import taichi_three as t3
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.reader import read
from Utils.utils_visualization import draw_pd_pn_image, init_mesh, update_mesh


if __name__ == "__main__":
    if not os.path.exists("modelVisualization"):
        os.makedirs("modelVisualization")
    for root, dirs, files in os.walk("modelVisualization/"):
        for name in files:
            os.remove(os.path.join(root, name))

    case_info = read(5)
    n_particles = case_info['mesh'].num_vertices
    n_elements = 0
    if case_info['dim'] == 2:
        n_elements = case_info['mesh'].num_faces
    elif case_info['dim'] == 3:
        n_elements = case_info['mesh'].num_elements

    gui = None
    camera = None

    if case_info['dim'] == 2:
        gui = ti.GUI("Model Visualizer", (1024, 1024), background_color=0xf7f7f7)
    elif case_info['dim'] == 3:
        camera = t3.Camera()
        amb_light = t3.AmbientLight(0.5)
        dir_light = t3.Light(dir=[-0.8, -0.6, -1.0])
        scene = t3.Scene()
        scene.add_camera(camera)
        scene.add_light(amb_light)
        scene.add_light(dir_light)

        boundary_points, boundary_edges, boundary_triangles = case_info['boundary']

        model = t3.Model(t3.DynamicMesh(n_faces=len(boundary_triangles) * 2,
                                        n_pos=case_info['mesh'].num_vertices,
                                        n_nrm=len(boundary_triangles) * 2))
        model.mesh.n_faces[None] = len(boundary_triangles) * 2
        init_mesh(model.mesh, boundary_triangles)
        model.L2W[None] = case_info['init_transformation']
        scene.add_model(model)
        gui = ti.GUI('Model Visualizer', camera.res)

    while True:
        if case_info['dim'] == 2:
            draw_pd_pn_image(gui, "modelVisualization/mesh.png",
                            case_info['mesh'].vertices.astype(np.float64)[:, 0:2],
                            case_info['mesh'].vertices.astype(np.float64)[:, 0:2],
                            case_info['mesh_offset'], case_info['mesh_scale'], case_info['mesh'].faces, n_elements)
        elif case_info['dim'] == 3:
            gui.get_event(None)
            model.mesh.pos.from_numpy(case_info['mesh'].vertices.astype(np.float32))
            update_mesh(model.mesh)
            # print(model.mesh.pos)
            camera.from_mouse(gui)
            scene.render()
            gui.set_image(camera.img)
            gui.show()
