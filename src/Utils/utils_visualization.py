import taichi as ti
# import taichi_three as t3
import taichi_glsl as ts
import numpy as np


# PN result: Red (Top Layer)
# PD result: Blue (Bottom Layer)
def draw_pd_pn_image(gui, file_name_path,
                     pd_x, pn_x,
                     mesh_offset, mesh_scale,
                     vertices, n_elements,
                     use_video_manager=False,
                     video_manager=None):
    particle_pos_pn = (pn_x + mesh_offset) * mesh_scale
    particle_pos_pd = (pd_x + mesh_offset) * mesh_scale

    for i in range(n_elements):
        for j in range(3):
            a, b = vertices[i, j], vertices[i, (j + 1) % 3]
            # PD
            gui.line((particle_pos_pd[a][0], particle_pos_pd[a][1]),
                     (particle_pos_pd[b][0], particle_pos_pd[b][1]),
                     radius=1,
                     color=0xFF0000)
            # PN
            gui.line((particle_pos_pn[a][0], particle_pos_pn[a][1]),
                     (particle_pos_pn[b][0], particle_pos_pn[b][1]),
                     radius=1,
                     color=0x0000FF)
    if not use_video_manager:
        gui.show(file_name_path)
    else:
        video_manager.write_frame(gui.get_image())
        gui.show()


def draw_image(gui, file_name_path,
               input_x,
               mesh_offset, mesh_scale,
               vertices, n_elements,
               use_video_manager=False,
               video_manager=None,
               line_color=0xFF0000):
    particle_pos = (input_x + mesh_offset) * mesh_scale
    for i in range(n_elements):
        for j in range(3):
            a, b = vertices[i, j], vertices[i, (j + 1) % 3]
            gui.line((particle_pos[a][0], particle_pos[a][1]),
                     (particle_pos[b][0], particle_pos[b][1]),
                     radius=1,
                     color=line_color)
    if not use_video_manager:
        gui.show(file_name_path)
    else:
        video_manager.write_frame(gui.get_image())
        gui.show()


def set_3D_scene(scene, camera, model, case_info):
    raise NotImplementedError("set_3D_scene() is Not Implemented")
    # amb_light = t3.AmbientLight(0.5)
    # dir_light = t3.Light(dir=case_info['light_dir'])
    # # pt_light = t3.PointLight(pos=[0, 0, 10])
    # scene.add_camera(camera)
    # scene.add_light(amb_light)
    # scene.add_light(dir_light)
    # # scene.add_light(pt_light)
    # boundary_points, boundary_edges, boundary_triangles = case_info['boundary']
    # model.mesh.n_faces[None] = len(boundary_triangles) * 2
    # init_mesh(model.mesh, boundary_triangles)
    # model.L2W[None] = case_info['init_transformation']
    # scene.add_model(model)


@ti.kernel
def init_mesh(mesh: ti.template(), triangles: ti.ext_arr()):
    for i in range(mesh.n_faces[None] / 2):
        mesh.faces[2 * i] = [[triangles[i, 2], 0, 2 * i],
                             [triangles[i, 1], 0, 2 * i],
                             [triangles[i, 0], 0, 2 * i]]
        mesh.faces[2 * i + 1] = [[triangles[i, 0], 0, 2 * i + 1],
                                 [triangles[i, 1], 0, 2 * i + 1],
                                 [triangles[i, 2], 0, 2 * i + 1]]


@ti.kernel
def update_mesh(mesh: ti.template()):
    for i in range(mesh.n_faces[None] / 2):
        pos1, pos2, pos3 = mesh.pos[mesh.faces[2 * i][0, 0]], mesh.pos[mesh.faces[2 * i][1, 0]], mesh.pos[mesh.faces[2 * i][2, 0]]
        normal = -ts.cross(pos1 - pos2, pos1 - pos3).normalized()
        mesh.nrm[2 * i] = normal
        mesh.nrm[2 * i + 1] = -normal


# It's only for 3D.
@ti.kernel
def update_boundary_pos(pos: ti.template(),
                        boundary_pos: ti.ext_arr(),
                        boundary_tris: ti.ext_arr(),
                        boundary_tri_num: ti.int32):
    for tri_idx in range(boundary_tri_num):
        for tri_vert_idx in ti.static(range(3)):
            for dim_idx in ti.static(range(3)):
                boundary_pos[tri_idx, tri_vert_idx, dim_idx] = pos[boundary_tris[tri_idx, tri_vert_idx]][dim_idx]


@ti.kernel
def update_boundary_pos_np(pos: ti.ext_arr(),
                        boundary_pos: ti.ext_arr(),
                        boundary_tris: ti.ext_arr(),
                        boundary_tri_num: ti.int32):
    for tri_idx in range(boundary_tri_num):
        for tri_vert_idx in ti.static(range(3)):
            for dim_idx in ti.static(range(3)):
                boundary_pos[tri_idx, tri_vert_idx, dim_idx] = pos[boundary_tris[tri_idx, tri_vert_idx], dim_idx]


def update_boundary_mesh(mesh_pos, boundary_pos, case_info):
    boundary_points, boundary_edges, boundary_triangles = case_info['boundary']
    update_boundary_pos(mesh_pos, boundary_pos, boundary_triangles, case_info['boundary_tri_num'])


def update_boundary_mesh_np(mesh_pos, boundary_pos, case_info):
    boundary_points, boundary_edges, boundary_triangles = case_info['boundary']
    update_boundary_pos_np(mesh_pos, boundary_pos, boundary_triangles, case_info['boundary_tri_num'])


def output_3d_seq(pos, boundary_tri, file_path):
    f = open(file_path, 'w')
    for [x, y, z] in pos:
        f.write('v %.6f %.6f %.6f\n' % (x, y, z))
    for [p0, p1, p2] in boundary_tri:
        f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
    f.close()


# For 2D: Angle is counter-clock wise and it uses [1, 0] direction as its start direction.
# For 3D: It uses Spherical coordinate system with its origin at the [0, 0, 0].
#         Angle1 will be used to determine the angle between y axis and the final direction.
#         Angle2 will be used to determine the direction along the x-z plane with a start direction at [1, 0, 0].
#         Angle1(Theta) should be a scalar in [0, pi). Angle2(Phi) should be a scalar in the range of [0, 2 * pi].
def get_force_field(mag, angle1, angle2=0.0, dim=2):
    if dim == 2:
        x = mag * ti.cos(ts.pi / 180.0 * angle1)
        y = mag * ti.sin(ts.pi / 180.0 * angle1)
        return [x, y]
    elif dim == 3:
        if angle1 < 0.0 or angle1 > 180:
            raise Exception("Angle1 is incorrect. Incorrect Angle1 is {}".format(angle1))
        if angle2 < 0.0 or angle2 > 360:
            raise Exception("Angle2 is incorrect. Incorrect Angle2 is {}".format(angle2))
        radian1 = ts.pi / 180.0 * angle1
        radian2 = ts.pi / 180.0 * angle2
        x = mag * ti.sin(radian1) * ti.cos(radian2)
        y = mag * ti.sin(radian1) * ti.sin(radian2)
        z = mag * ti.cos(radian2)
        return [x, y, z]
    else:
        raise Exception("Force field dim doesn't correct. Error dim is {}".format(dim))


def rotate_matrix_y_axis(beta_degree):
    beta_radian = np.radians(beta_degree)
    return np.array([[np.cos(beta_radian), 0.0, -np.sin(beta_radian), 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [np.sin(beta_radian), 0.0, np.cos(beta_radian), 0.0],
                     [0.0, 0.0, 0.0, 1.0]])



