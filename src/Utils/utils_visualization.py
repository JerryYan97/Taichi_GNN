import taichi as ti
# import taichi_three as t3
import taichi_glsl as ts
import numpy as np
import math
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.linalg import polar


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
def get_acc_field(mag, angle1, angle2=0.0, dim=2):
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
        z = mag * ti.cos(radian1)
        return [x, y, z]
    else:
        raise Exception("Acc field dim doesn't correct. Error dim is {}".format(dim))


def rotate_matrix_y_axis(beta_degree):
    beta_radian = np.radians(beta_degree)
    return np.array([[np.cos(beta_radian), 0.0, -np.sin(beta_radian), 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [np.sin(beta_radian), 0.0, np.cos(beta_radian), 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


# Tait-Bryan angles
def rotate_general(alpha_deg, beta_deg, gamma_deg):
    alpha_rad = np.radians(alpha_deg)
    beta_rad = np.radians(beta_deg)
    gamma_rad = np.radians(gamma_deg)
    return np.array([[np.cos(alpha_rad) * np.cos(beta_rad),
                      np.cos(alpha_rad) * np.sin(beta_rad) * np.sin(gamma_rad) - np.sin(alpha_rad) * np.cos(gamma_rad),
                      np.cos(alpha_rad) * np.sin(beta_rad) * np.cos(gamma_rad) + np.sin(alpha_rad) * np.sin(gamma_rad),
                      0.0],
                     [np.sin(alpha_rad) * np.cos(beta_rad),
                      np.sin(alpha_rad) * np.sin(beta_rad) * np.sin(gamma_rad) + np.cos(alpha_rad) * np.cos(gamma_rad),
                      np.sin(alpha_rad) * np.sin(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad) * np.sin(gamma_rad),
                      0.0],
                     [-np.sin(beta_rad), np.cos(beta_rad) * np.sin(gamma_rad),
                      np.cos(beta_rad) * np.cos(gamma_rad), 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

# mag: magnitude of the Acceleration
# center: bbox center of the mesh
# pos: the mesh point position that want to get the acceleration
# angle: default: along x axis, used to change the acceleration field
# width: larger than this value, zero acceleration
@ti.func
def get_ring_acc_field(mag, width, center, pos, angle, dim) -> ti.Vector:
    radian = math.pi / 180.0 * angle
    p1 = center
    p2 = center + ti.Vector([0.0, 1.0*width, 0.0])
    p3 = center + ti.Vector([ti.cos(radian)*width, 0.0, ti.sin(radian)*width])
    a = (p2[1]-p1[1])*(p3[2]-p1[2])-(p3[1]-p1[1])*(p2[2]-p1[2])
    b = (p2[2]-p1[2])*(p3[0]-p1[0])-(p3[2]-p1[2])*(p2[0]-p1[0])
    c = (p2[0]-p1[0])*(p3[1]-p1[1])-(p3[0]-p1[0])*(p2[1]-p1[1])
    d = -a*p1[0]-b*p1[1]-c*p1[2]
    t = (a*pos[0]+b*pos[1]+c*pos[2]+d)/(a*a+b*b+c*c)
    p = ti.Vector([pos[0]-a*t, pos[1]-b*t, pos[2]-c*t])
    T = ti.Vector([0.0, 0.0, 0.0])
    if (p-pos).norm() <= width:
        l = (p-center).norm()
        L = l * l / ti.sqrt((p[0]-center[0])*(p[0]-center[0])+(p[2]-center[2])*(p[2]-center[2]))
        ss = (p3-center).dot((p-center))                        # print("l: ", l, "L: ", L)
        p4 = center + (ss * (p3 - center)).normalized() * L     # print("p4: ", p4)
        s = (p4-p).dot(p3-center) * p[1]
        if ti.abs(p[1]) < 0.000000001:
            T = (ss * ti.Vector([0.0, -1.0, 0.0])).normalized()
            T = mag * T
        else:
            T = (s*(p4-p)).normalized()                         # print("p4-p3: ", p4-p3)
            T = mag * T
    return T


@ti.func
def get_ring_circle_acc_field(mag, width, center, pos, angle, min_radius, dim) -> ti.Vector:
    radian = math.pi / 180.0 * angle
    p1 = center
    p2 = center + ti.Vector([0.0, 1.0*width, 0.0])
    p3 = center + ti.Vector([ti.cos(radian)*width, 0.0, ti.sin(radian)*width])
    a = (p2[1]-p1[1])*(p3[2]-p1[2])-(p3[1]-p1[1])*(p2[2]-p1[2])
    b = (p2[2]-p1[2])*(p3[0]-p1[0])-(p3[2]-p1[2])*(p2[0]-p1[0])
    c = (p2[0]-p1[0])*(p3[1]-p1[1])-(p3[0]-p1[0])*(p2[1]-p1[1])
    d = -a*p1[0]-b*p1[1]-c*p1[2]
    t = (a*pos[0]+b*pos[1]+c*pos[2]+d)/(a*a+b*b+c*c)
    p = ti.Vector([pos[0]-a*t, pos[1]-b*t, pos[2]-c*t])
    T = ti.Vector([0.0, 0.0, 0.0])
    if (p-pos).norm() <= width and (p-center).norm() > min_radius:
        l = (p-center).norm()
        L = l * l / ti.sqrt((p[0]-center[0])*(p[0]-center[0])+(p[2]-center[2])*(p[2]-center[2]))
        ss = (p3-center).dot((p-center))                        # print("l: ", l, "L: ", L)
        p4 = center + (ss * (p3 - center)).normalized() * L     # print("p4: ", p4)
        s = (p4-p).dot(p3-center) * p[1]
        if ti.abs(p[1]) < 0.000000001:
            T = (ss * ti.Vector([0.0, -1.0, 0.0])).normalized()
            T = mag * T
        else:
            T = (s*(p4-p)).normalized()                          # print("p4-p3: ", p4-p3)
            T = mag * T
    return T


@ti.func
def get_point_acc_field(min, max, pos, acc) -> ti.Vector:
    T = ti.Vector([0.0, 0.0, 0.0])
    if min[0] < pos[0] < max[0] and min[1] < pos[1] < max[1] and min[2] < pos[2] < max[2]:
        T = acc
    else:
        T = T
    return T


@ti.func
def get_point_acc_field_by_point(t_pos, pos, radius, acc) -> ti.Vector:
    T = ti.Vector([0.0, 0.0, 0.0])
    if (t_pos - pos).norm() < radius:
        T = acc
    else:
        T = T
    return T


@ti.func
def get_point_acc_field_ref(min, max, pos, acc, ti_acc) -> ti.Vector:
    if min[0] < pos[0] < max[0] and min[1] < pos[1] < max[1] and min[2] < pos[2] < max[2]:
        ti_acc = acc
    else:
        ti_acc = ti.Vector([0.0, 0.0, 0.0])


@ti.func
def RM2Euler_ti(R) -> ti.Vector:  # matrix no type hint needed
    theta_x = ti.atan2(R[2, 1], R[2, 2])
    theta_y = ti.atan2(-R[2, 0], ti.sqrt(R[2, 1]*R[2, 1]+R[2, 2]*R[2, 2]))
    theta_z = ti.atan2(R[2, 0], R[0, 0])
    return ti.Vector([theta_x, theta_y, theta_z])


def calcR(A_pq):
    S = sqrtm(np.dot(np.transpose(A_pq), A_pq))
    R = np.dot(A_pq, inv(S))
    return R, S


def polarR(M):
    R, S = polar(M)
    return R, S


def RM2Euler_radian(A):  # numpy
    R, S = calcR(A)
    theta_x = np.arctan2(R[2, 1], R[2, 2])
    theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]*R[2, 1]+R[2, 2]*R[2, 2]))
    theta_z = np.arctan2(R[2, 0], R[0, 0])
    return np.array([theta_x, theta_y, theta_z]), np.array([S[0, 0], S[1, 1], S[2, 2]])


def RM2Euler(A):  # numpy
    # R, S = calcR(A)
    R, S = polar(A)
    theta_x = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
    theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]*R[2, 1]+R[2, 2]*R[2, 2])) * 180 / np.pi
    theta_z = np.arctan2(R[2, 0], R[0, 0]) * 180 / np.pi
    return np.array([theta_x, theta_y, theta_z]), np.array([S[0, 0], S[1, 1], S[2, 2]])


