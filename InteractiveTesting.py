import tina
import taichi as ti
from src.Utils.reader import read
import numpy as np
from numpy import linalg as LA


@ti.kernel
def raySphereIntersect(ray_dir: ti.ext_arr(), ray_origin: ti.ext_arr(),
                       s_center: ti.ext_arr(), dist_array: ti.ext_arr(), vert_num: ti.i32):
    radius = 0.3
    for i in range(vert_num):
        ray_dir_ti = ti.Vector([ray_dir[0], ray_dir[1], ray_dir[2]])
        ray_center_ti = ti.Vector([ray_origin[0], ray_origin[1], ray_origin[2]])
        s_center_ti = ti.Vector([s_center[i, 0], s_center[i, 1], s_center[i, 2]])
        l1 = s_center_ti - ray_center_ti
        l1_len = l1.norm()
        l2_len = l1.dot(ray_dir_ti)
        if l2_len > 0.0:
            # Same direction
            l3_len = ti.sqrt(l1_len ** 2 - l2_len ** 2)
            if l3_len <= radius:
                # Hit
                dist_array[i] = l3_len


if __name__ == "__main__":
    test_case = 1011
    case_info = read(test_case)
    mesh = case_info['mesh']

    ti.init(ti.gpu)
    scene = tina.Scene()
    pars = tina.SimpleParticles()
    material = tina.BlinnPhong()
    scene.add_object(pars, material)

    gui = ti.GUI('kmeans visualization')
    pars.set_particles(mesh.vertices)
    pars.set_particle_radii(np.full(mesh.num_vertices, 0.05))
    particles_color_base = np.full((mesh.num_vertices, 3), 1.0, dtype=float)
    lock_gui_flag = True

    while gui.running:
        scene.input(gui)
        cam_pos = scene.control.center + scene.control.back
        particles_color_offset = np.full((mesh.num_vertices, 3), 0.0, dtype=float)

        if gui.is_pressed(ti.GUI.LMB):
            # Select
            mouse_x, mouse_y = gui.get_cursor_pos()
            # print("mouse x:", mouse_x, " mouse y:", mouse_y)
            relative_x = mouse_x - 0.5
            relative_y = mouse_y - 0.5
            sp_x = mouse_x * 2.0 - 1.0
            sp_y = mouse_y * 2.0 - 1.0
            sp_pos = np.array([sp_x, sp_y, 0.0, 1.0])
            V2W_np = scene.engine.V2W.to_numpy()
            wp_pos = V2W_np @ (sp_pos * 0.2)
            # print("wp_pos:", wp_pos)
            ray_dir = wp_pos[0:3] - cam_pos
            ray_dir /= LA.norm(ray_dir)
            dist_arr = np.ones(mesh.num_vertices, dtype=float) * 100000.0
            raySphereIntersect(ray_dir, cam_pos, mesh.vertices, dist_arr, mesh.num_vertices)
            min_idx = np.argmin(dist_arr)
            min_val = np.amin(dist_arr)
            # print("min idx:", min_idx)
            # print("min val:", min_val)

            if min_val != 100000.0:
                particles_color_offset[min_idx, 1] = -1.0
                particles_color_offset[min_idx, 2] = -1.0
            pars.set_particle_colors(particles_color_offset + particles_color_base)

        elif gui.is_pressed('d', ti.GUI.RIGHT):
            print('Go right!')
        elif gui.is_pressed('l'):
            print('Lock GUI')
            lock_gui_flag = False
        elif gui.is_pressed('u'):
            print('Unlock GUI')
            lock_gui_flag = True

        scene.render()
        gui.set_image(scene.img)
        gui.show()



