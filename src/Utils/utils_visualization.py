import taichi as ti
# import taichi_three as t3
import taichi_glsl as ts


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
    amb_light = t3.AmbientLight(0.5)
    dir_light = t3.Light(dir=case_info['light_dir'])
    # pt_light = t3.PointLight(pos=[0, 0, 10])
    scene.add_camera(camera)
    scene.add_light(amb_light)
    scene.add_light(dir_light)
    # scene.add_light(pt_light)
    boundary_points, boundary_edges, boundary_triangles = case_info['boundary']
    model.mesh.n_faces[None] = len(boundary_triangles) * 2
    init_mesh(model.mesh, boundary_triangles)
    model.L2W[None] = case_info['init_transformation']
    scene.add_model(model)


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

    # for tri_idx in range(boundary_tri_num):
    #     b_pos_idx1, b_pos_idx2, b_pos_idx3 = tri_idx * 3, tri_idx * 3 + 1, tri_idx * 3 + 2
    #     tri_pos_idx1 = boundary_tris[tri_idx, 0]
    #     tri_pos_idx2 = boundary_tris[tri_idx, 1]
    #     tri_pos_idx3 = boundary_tris[tri_idx, 2]
    #     for dim_idx in ti.static(range(3)):
    #         boundary_pos[b_pos_idx1, dim_idx] = pos[tri_pos_idx1][dim_idx]
    #         boundary_pos[b_pos_idx2, dim_idx] = pos[tri_pos_idx2][dim_idx]
    #         boundary_pos[b_pos_idx3, dim_idx] = pos[tri_pos_idx3][dim_idx]


# For 2D: Angle is counter-clock wise and it uses [1, 0] direction as its start direction.
# For 3D: It uses Spherical coordinate system with its origin at the [0, 0, 0].
#         Angle1 will be used to determine the angle between y axis and the final direction.
#         Angle2 will be used to determine the direction along the x-z plane with a start direction at [1, 0, 0].
#         Angle1(Theta) should be a scalar in [0, 2 * pi). Angle2(Phi) should be a scalar in the range of [0, pi].
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


# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
