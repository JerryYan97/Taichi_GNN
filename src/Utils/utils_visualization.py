import taichi as ti
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
