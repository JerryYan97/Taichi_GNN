import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import taichi as ti
# import taichi_three as t3
import numpy as np
from Utils.JGSL_WATER import *
from Utils.neo_hookean import fixed_corotated_first_piola_kirchoff_stress
from Utils.neo_hookean import fixed_corotated_energy
from Utils.neo_hookean import fixed_corotated_first_piola_kirchoff_stress_derivative
from Utils.reader import read
from Utils.math_tools import svd, my_svd
from Utils.utils_visualization import draw_image, update_boundary_mesh, output_3d_seq, get_acc_field, get_ring_acc_field, get_point_acc_field_by_point

##############################################################################
case_info = read(1011)
mesh = case_info['mesh']
dirichlet = case_info['dirichlet']
mesh_scale = case_info['mesh_scale']
mesh_offset = case_info['mesh_offset']
dim = case_info['dim']
center = case_info['center']
min_sphere_radius = case_info['min_sphere_radius']
##############################################################################
_, _, boundary_triangles = case_info['boundary']

ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False)

real = ti.f64

dt = 0.01
E = 5e6
nu = 0.4
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
density = 1e3
n_particles = mesh.num_vertices
if dim == 2:
    n_elements = mesh.num_faces
else:
    n_elements = mesh.num_elements
    import tina
    scene = tina.Scene(culling=False, clipping=True)
    tina_mesh = tina.SimpleMesh()
    model = tina.MeshTransform(tina_mesh)
    scene.add_object(model)
    boundary_pos = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)

cnt = ti.field(ti.i32, shape=())

x = ti.Vector.field(dim, real, n_particles)
xPrev = ti.Vector.field(dim, real, n_particles)
xTilde = ti.Vector.field(dim, real, n_particles)
xn = ti.Vector.field(dim, real, n_particles)
v = ti.Vector.field(dim, real, n_particles)
m = ti.field(real, n_particles)

ti_vel_last = ti.Vector.field(dim, real, n_particles)
ti_vel_del = ti.Vector.field(dim, real, n_particles)

zero = ti.Vector.field(dim, real, n_particles)
restT = ti.Matrix.field(dim, dim, real, n_elements)
vertices = ti.field(ti.i32, (n_elements, dim + 1))

# Complie time optimization data structure
ti_dPdF_field = ti.field(real, (n_elements, dim * dim, dim * dim))
ti_intermediate_field = ti.field(real, (n_elements, dim * (dim + 1), dim * dim))
ti_M_field = ti.field(real, (n_elements, dim * dim, dim * dim))
ti_U_field = ti.field(real, (n_elements, dim * dim, dim * dim))
ti_V_field = ti.field(real, (n_elements, dim * dim, dim * dim))
ti_indMap_field = ti.field(real, (n_elements, dim * (dim + 1)))

# external acc -- Angle: from [1, 0] -- counter-clock wise
ex_acc = ti.Vector.field(dim, real, n_particles)
ti_center = ti.Vector([center[0], center[1], center[2]])


def initial():
    x.from_numpy(mesh.vertices.astype(np.float64))
    if dim == 2:
        vertices.from_numpy(mesh.faces)
    if dim == 3:
        vertices.from_numpy(mesh.elements)
    xPrev.fill(0)
    xTilde.fill(0)
    xn.fill(0)
    v.fill(0)
    m.fill(0)
    zero.fill(0)
    restT.fill(0)

    ti_vel_last.fill(0)
    ti_vel_del.fill(0)
    ################################ external accleration ######################################
    ex_acc.fill(0)


@ti.kernel
def set_dir_acc_3D():
    if ti.static(dim == 2):
        for i in range(n_particles):
            ex_acc[i] = ti.Vector(get_acc_field(6, -45.0))
    else:
        # exf_mag = 0.0002 # 1001: 6   1003 and 1004: 0.06  1005: 0.0002
        for i in range(n_particles):
            # ex_acc[i] = ti.Vector(get_acc_field(2.0, 45.0, 45.0, 3))
            # ex_acc[i] = ti.Vector([0.0, -12.0, 12.0])
            ex_acc[i] = ti.Vector([0.0, -980.0, 0.0])


@ti.kernel
def set_ring_acc_3D():
    for i in range(n_particles):
        ex_acc[i] = get_ring_acc_field(0.04, 10.0, ti_center, x[i], 0.0, 3)


@ti.kernel
def set_point_acc_by_point_3D(pf_ind: ti.i32, pf_radius: ti.f64, xx: ti.f32, y: ti.f32, z: ti.f32):
    for i in range(n_particles):  # t_pos, pos, radius, acc
        ex_acc[i] = get_point_acc_field_by_point(x[pf_ind], x[i], pf_radius, ti.Vector([xx, y, z]))


@ti.func
def compute_T(i):
    if ti.static(dim == 2):
        ab = x[vertices[i, 1]] - x[vertices[i, 0]]
        ac = x[vertices[i, 2]] - x[vertices[i, 0]]
        return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])
    else:
        ab = x[vertices[i, 1]] - x[vertices[i, 0]]
        ac = x[vertices[i, 2]] - x[vertices[i, 0]]
        ad = x[vertices[i, 3]] - x[vertices[i, 0]]
        return ti.Matrix([[ab[0], ac[0], ad[0]], [ab[1], ac[1], ad[1]], [ab[2], ac[2], ad[2]]])


@ti.kernel
def compute_restT_and_m():
    for i in range(n_elements):
        restT[i] = compute_T(i)
        mass = restT[i].determinant() / dim / (dim - 1) * density / (dim + 1)
        if mass < 0.0:
            print("FATAL ERROR : mesh inverted")
        for d in ti.static(range(dim + 1)):
            m[vertices[i, d]] += mass


@ti.kernel
def compute_xn_and_xTilde():
    if ti.static(dim == 2):
        for i in range(n_particles):
            xn[i] = x[i]
            xTilde[i] = x[i] + dt * v[i]
            xTilde(0)[i] += dt * dt * (ex_acc[i][0])
            xTilde(1)[i] += dt * dt * (ex_acc[i][1])
    if ti.static(dim == 3):
        for i in range(n_particles):
            xn[i] = x[i]
            xTilde[i] = x[i] + dt * v[i]
            xTilde(0)[i] += dt * dt * (ex_acc[i][0])
            xTilde(1)[i] += dt * dt * (ex_acc[i][1])
            xTilde(2)[i] += dt * dt * (ex_acc[i][2])


@ti.kernel
def check_acceleration_status() -> ti.f64:
    residual = 0.0
    for i in range(n_particles):
        residual += (ti_vel_del[i]/dt).norm()
    residual /= (1.0 * n_particles)
    print("acceleration :", residual)
    return residual


@ti.kernel
def compute_energy() -> real:
    total_energy = 0.0
    # inertia
    for i in range(n_particles):
        total_energy += 0.5 * m[i] * (x[i] - xTilde[i]).norm_sqr()
    # elasticity
    for e in range(n_elements):
        F = compute_T(e) @ restT[e].inverse()
        vol0 = restT[e].determinant() / dim / (dim - 1)
        # U, sig, V = svd(F)
        U, sig, V = my_svd(F)
        total_energy += fixed_corotated_energy(sig, la, mu) * dt * dt * vol0
    return total_energy


@ti.kernel
def compute_hessian_and_gradient(data_mat: ti.ext_arr(), data_rhs: ti.ext_arr()):
    cnt[None] = 0
    # inertia
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            c = cnt[None] + i * dim + d   # 2 * n_p
            data_mat[0, c] = i * dim + d
            data_mat[1, c] = i * dim + d
            data_mat[2, c] = m[i]
            data_rhs[i * dim + d] -= m[i] * (x(d)[i] - xTilde(d)[i])
    cnt[None] += n_particles * dim
    # elasticity
    for e in range(n_elements):
        F = compute_T(e) @ restT[e].inverse()  # 2 * 2
        IB = restT[e].inverse()  # 2 * 2
        vol0 = restT[e].determinant() / dim / (dim - 1)
        fixed_corotated_first_piola_kirchoff_stress_derivative(F, la, mu, ti_dPdF_field, ti_M_field, ti_U_field, ti_V_field, e, dt, vol0)
        P = fixed_corotated_first_piola_kirchoff_stress(F, la, mu) * dt * dt * vol0
        if ti.static(dim == 2):
            for colI in range(4):
                _000 = ti_dPdF_field[e, 0, colI] * IB[0, 0]
                _010 = ti_dPdF_field[e, 0, colI] * IB[1, 0]
                _101 = ti_dPdF_field[e, 2, colI] * IB[0, 1]
                _111 = ti_dPdF_field[e, 2, colI] * IB[1, 1]
                _200 = ti_dPdF_field[e, 1, colI] * IB[0, 0]
                _210 = ti_dPdF_field[e, 1, colI] * IB[1, 0]
                _301 = ti_dPdF_field[e, 3, colI] * IB[0, 1]
                _311 = ti_dPdF_field[e, 3, colI] * IB[1, 1]
                ti_intermediate_field[e, 2, colI] = _000 + _101
                ti_intermediate_field[e, 3, colI] = _200 + _301
                ti_intermediate_field[e, 4, colI] = _010 + _111
                ti_intermediate_field[e, 5, colI] = _210 + _311
                ti_intermediate_field[e, 0, colI] = -ti_intermediate_field[e, 2, colI] - ti_intermediate_field[e, 4, colI]
                ti_intermediate_field[e, 1, colI] = -ti_intermediate_field[e, 3, colI] - ti_intermediate_field[e, 5, colI]

            ti_indMap_field[e, 0] = vertices[e, 0] * 2
            ti_indMap_field[e, 1] = vertices[e, 0] * 2 + 1
            ti_indMap_field[e, 2] = vertices[e, 1] * 2
            ti_indMap_field[e, 3] = vertices[e, 1] * 2 + 1
            ti_indMap_field[e, 4] = vertices[e, 2] * 2
            ti_indMap_field[e, 5] = vertices[e, 2] * 2 + 1

            for colI in range(6):
                _000 = ti_intermediate_field[e, colI, 0] * IB[0, 0]
                _010 = ti_intermediate_field[e, colI, 0] * IB[1, 0]
                _101 = ti_intermediate_field[e, colI, 2] * IB[0, 1]
                _111 = ti_intermediate_field[e, colI, 2] * IB[1, 1]
                _200 = ti_intermediate_field[e, colI, 1] * IB[0, 0]
                _210 = ti_intermediate_field[e, colI, 1] * IB[1, 0]
                _301 = ti_intermediate_field[e, colI, 3] * IB[0, 1]
                _311 = ti_intermediate_field[e, colI, 3] * IB[1, 1]
                c = cnt[None] + e * 36 + colI * 6 + 0
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, 2], ti_indMap_field[e, colI], _000 + _101
                c = cnt[None] + e * 36 + colI * 6 + 1
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, 3], ti_indMap_field[e, colI], _200 + _301
                c = cnt[None] + e * 36 + colI * 6 + 2
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, 4], ti_indMap_field[e, colI], _010 + _111
                c = cnt[None] + e * 36 + colI * 6 + 3
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, 5], ti_indMap_field[e, colI], _210 + _311
                c = cnt[None] + e * 36 + colI * 6 + 4
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, 0], ti_indMap_field[e, colI], - _000 - _101 - _010 - _111
                c = cnt[None] + e * 36 + colI * 6 + 5
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, 1], ti_indMap_field[e, colI], - _200 - _301 - _210 - _311
            data_rhs[vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
            data_rhs[vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
            data_rhs[vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
            data_rhs[vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
            data_rhs[vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
            data_rhs[vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]
        else:
            for colI in range(9):
                ti_intermediate_field[e, 3, colI] = IB[0, 0] * ti_dPdF_field[e, 0, colI] + IB[0, 1] * ti_dPdF_field[e, 3, colI] + IB[0, 2] * ti_dPdF_field[e, 6, colI]
                ti_intermediate_field[e, 4, colI] = IB[0, 0] * ti_dPdF_field[e, 1, colI] + IB[0, 1] * ti_dPdF_field[e, 4, colI] + IB[0, 2] * ti_dPdF_field[e, 7, colI]
                ti_intermediate_field[e, 5, colI] = IB[0, 0] * ti_dPdF_field[e, 2, colI] + IB[0, 1] * ti_dPdF_field[e, 5, colI] + IB[0, 2] * ti_dPdF_field[e, 8, colI]
                ti_intermediate_field[e, 6, colI] = IB[1, 0] * ti_dPdF_field[e, 0, colI] + IB[1, 1] * ti_dPdF_field[e, 3, colI] + IB[1, 2] * ti_dPdF_field[e, 6, colI]
                ti_intermediate_field[e, 7, colI] = IB[1, 0] * ti_dPdF_field[e, 1, colI] + IB[1, 1] * ti_dPdF_field[e, 4, colI] + IB[1, 2] * ti_dPdF_field[e, 7, colI]
                ti_intermediate_field[e, 8, colI] = IB[1, 0] * ti_dPdF_field[e, 2, colI] + IB[1, 1] * ti_dPdF_field[e, 5, colI] + IB[1, 2] * ti_dPdF_field[e, 8, colI]
                ti_intermediate_field[e, 9, colI] = IB[2, 0] * ti_dPdF_field[e, 0, colI] + IB[2, 1] * ti_dPdF_field[e, 3, colI] + IB[2, 2] * ti_dPdF_field[e, 6, colI]
                ti_intermediate_field[e, 10, colI] = IB[2, 0] * ti_dPdF_field[e, 1, colI] + IB[2, 1] * ti_dPdF_field[e, 4, colI] + IB[2, 2] * ti_dPdF_field[e, 7, colI]
                ti_intermediate_field[e, 11, colI] = IB[2, 0] * ti_dPdF_field[e, 2, colI] + IB[2, 1] * ti_dPdF_field[e, 5, colI] + IB[2, 2] * ti_dPdF_field[e, 8, colI]
                ti_intermediate_field[e, 0, colI] = -ti_intermediate_field[e, 3, colI] - ti_intermediate_field[e, 6, colI] - ti_intermediate_field[e, 9, colI]
                ti_intermediate_field[e, 1, colI] = -ti_intermediate_field[e, 4, colI] - ti_intermediate_field[e, 7, colI] - ti_intermediate_field[e, 10, colI]
                ti_intermediate_field[e, 2, colI] = -ti_intermediate_field[e, 5, colI] - ti_intermediate_field[e, 8, colI] - ti_intermediate_field[e, 11, colI]

            ti_indMap_field[e, 0] = vertices[e, 0] * 3
            ti_indMap_field[e, 1] = vertices[e, 0] * 3 + 1
            ti_indMap_field[e, 2] = vertices[e, 0] * 3 + 2
            ti_indMap_field[e, 3] = vertices[e, 1] * 3
            ti_indMap_field[e, 4] = vertices[e, 1] * 3 + 1
            ti_indMap_field[e, 5] = vertices[e, 1] * 3 + 2
            ti_indMap_field[e, 6] = vertices[e, 2] * 3
            ti_indMap_field[e, 7] = vertices[e, 2] * 3 + 1
            ti_indMap_field[e, 8] = vertices[e, 2] * 3 + 2
            ti_indMap_field[e, 9] = vertices[e, 3] * 3
            ti_indMap_field[e, 10] = vertices[e, 3] * 3 + 1
            ti_indMap_field[e, 11] = vertices[e, 3] * 3 + 2

            for rowI in range(12):
                c = cnt[None] + e * 144 + rowI * 12 + 0
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 3], IB[0, 0] * ti_intermediate_field[e, rowI, 0] + IB[0, 1] * ti_intermediate_field[e, rowI, 3] + IB[0, 2] * ti_intermediate_field[e, rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 1
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 4], IB[0, 0] * ti_intermediate_field[e, rowI, 1] + IB[0, 1] * ti_intermediate_field[e, rowI, 4] + IB[0, 2] * ti_intermediate_field[e, rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 2
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 5], IB[0, 0] * ti_intermediate_field[e, rowI, 2] + IB[0, 1] * ti_intermediate_field[e, rowI, 5] + IB[0, 2] * ti_intermediate_field[e, rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 3
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 6], IB[1, 0] * ti_intermediate_field[e, rowI, 0] + IB[1, 1] * ti_intermediate_field[e, rowI, 3] + IB[1, 2] * ti_intermediate_field[e, rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 4
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 7], IB[1, 0] * ti_intermediate_field[e, rowI, 1] + IB[1, 1] * ti_intermediate_field[e, rowI, 4] + IB[1, 2] * ti_intermediate_field[e, rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 5
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 8], IB[1, 0] * ti_intermediate_field[e, rowI, 2] + IB[1, 1] * ti_intermediate_field[e, rowI, 5] + IB[1, 2] * ti_intermediate_field[e, rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 6
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 9], IB[2, 0] * ti_intermediate_field[e, rowI, 0] + IB[2, 1] * ti_intermediate_field[e, rowI, 3] + IB[2, 2] * ti_intermediate_field[e, rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 7
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 10], IB[2, 0] * ti_intermediate_field[e, rowI, 1] + IB[2, 1] * ti_intermediate_field[e, rowI, 4] + IB[2, 2] * ti_intermediate_field[e, rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 8
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 11], IB[2, 0] * ti_intermediate_field[e, rowI, 2] + IB[2, 1] * ti_intermediate_field[e, rowI, 5] + IB[2, 2] * ti_intermediate_field[e, rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 9
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 0], -data_mat[2, c - 9] - data_mat[2, c - 6] - data_mat[2, c - 3]
                c = cnt[None] + e * 144 + rowI * 12 + 10
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 1], -data_mat[2, c - 9] - data_mat[2, c - 6] - data_mat[2, c - 3]
                c = cnt[None] + e * 144 + rowI * 12 + 11
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = ti_indMap_field[e, rowI], ti_indMap_field[e, 2], -data_mat[2, c - 9] - data_mat[2, c - 6] - data_mat[2, c - 3]
            R10 = IB[0, 0] * P[0, 0] + IB[0, 1] * P[0, 1] + IB[0, 2] * P[0, 2]
            R11 = IB[0, 0] * P[1, 0] + IB[0, 1] * P[1, 1] + IB[0, 2] * P[1, 2]
            R12 = IB[0, 0] * P[2, 0] + IB[0, 1] * P[2, 1] + IB[0, 2] * P[2, 2]
            R20 = IB[1, 0] * P[0, 0] + IB[1, 1] * P[0, 1] + IB[1, 2] * P[0, 2]
            R21 = IB[1, 0] * P[1, 0] + IB[1, 1] * P[1, 1] + IB[1, 2] * P[1, 2]
            R22 = IB[1, 0] * P[2, 0] + IB[1, 1] * P[2, 1] + IB[1, 2] * P[2, 2]
            R30 = IB[2, 0] * P[0, 0] + IB[2, 1] * P[0, 1] + IB[2, 2] * P[0, 2]
            R31 = IB[2, 0] * P[1, 0] + IB[2, 1] * P[1, 1] + IB[2, 2] * P[1, 2]
            R32 = IB[2, 0] * P[2, 0] + IB[2, 1] * P[2, 1] + IB[2, 2] * P[2, 2]
            data_rhs[vertices[e, 1] * 3 + 0] -= R10
            data_rhs[vertices[e, 1] * 3 + 1] -= R11
            data_rhs[vertices[e, 1] * 3 + 2] -= R12
            data_rhs[vertices[e, 2] * 3 + 0] -= R20
            data_rhs[vertices[e, 2] * 3 + 1] -= R21
            data_rhs[vertices[e, 2] * 3 + 2] -= R22
            data_rhs[vertices[e, 3] * 3 + 0] -= R30
            data_rhs[vertices[e, 3] * 3 + 1] -= R31
            data_rhs[vertices[e, 3] * 3 + 2] -= R32
            data_rhs[vertices[e, 0] * 3 + 0] -= -R10 - R20 - R30
            data_rhs[vertices[e, 0] * 3 + 1] -= -R11 - R21 - R31
            data_rhs[vertices[e, 0] * 3 + 2] -= -R12 - R22 - R32
    cnt[None] += n_elements * (dim + 1) * dim * (dim + 1) * dim


@ti.kernel
def save_xPrev():
    for i in range(n_particles):
        xPrev[i] = x[i]


@ti.kernel
def apply_sol(alpha : real, data_sol: ti.ext_arr()):
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            x(d)[i] = xPrev(d)[i] + data_sol[i * dim + d] * alpha


@ti.kernel
def compute_v():
    for i in range(n_particles):
        v[i] = (x[i] - xn[i]) / dt
        ti_vel_del[i] = v[i] - ti_vel_last[i]
        ti_vel_last[i] = v[i]


@ti.kernel
def output_residual(data_sol: ti.ext_arr()) -> real:
    residual = 0.0
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            residual = ti.max(residual, ti.abs(data_sol[i * dim + d]))
    # print("Search Direction Residual : ", residual / dt)
    return residual


@ti.kernel
def output_residual2(data_sol: ti.ext_arr()) -> real:
    residual = 0.0
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            residual += data_sol[i * dim + d] * data_sol[i * dim + d]
    residual = ti.sqrt(residual)
    print("Search Direction Residual : ", residual)
    return residual


def output_aux_data(f):
    if dim == 3:
        name_pn = "./AnimSeq/PNAnimSeq/PN_pbpF_" + case_info['case_name'] + "_" + str(f).zfill(6) + ".obj"
        output_3d_seq(x.to_numpy(), boundary_triangles, name_pn)


def animation_control(f):
    # Set force
    # if f == 50:
    #     ex_acc.fill(0)
    pass


@ti.kernel
def pos_init():
    for i in range(n_particles):
        x[i] = ti.Vector([float(ti.random()) * 0.05, float(ti.random()) * 0.05, float(ti.random()) * 0.05])


def animation_init():
    # Init pos
    pos_init()
    # Init vec


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for root, dirs, files in os.walk("results/"):
        for name in files:
            os.remove(os.path.join(root, name))
    os.makedirs("AnimSeq", exist_ok=True)
    os.makedirs("./AnimSeq/PNAnimSeq", exist_ok=True)
    for root, dirs, files in os.walk("./AnimSeq/PNAnimSeq/"):
        for name in files:
            os.remove(os.path.join(root, name))

    initial()
    compute_restT_and_m()
    stop_acceleration = 0.001

    video_manager = ti.VideoManager(output_dir=os.getcwd() + '/results/', framerate=24, automatic_build=False)
    frame_counter = 0

    if dim == 2:
        gui = ti.GUI('PN Standalone', background_color=0xf7f7f7)
        filename = f'./results/frame_rest.png'
        draw_image(gui, filename, x.to_numpy(), mesh_offset, mesh_scale, vertices.to_numpy(), n_elements)
    else:
        gui = ti.GUI('PN standalone 3D')
        # model.set_transform(case_info['transformation_mat'])
        update_boundary_mesh(x, boundary_pos, case_info)
        scene.input(gui)
        tina_mesh.set_face_verts(boundary_pos)
        scene.render()
        gui.set_image(scene.img)
        gui.show()

    data_rhs = np.zeros(shape=(200000,), dtype=np.float64)
    data_mat = np.zeros(shape=(3, 20000000), dtype=np.float64)
    data_sol = np.zeros(shape=(200000,), dtype=np.float64)

    mag = 4.6
    # set_point_acc_by_point_3D(1, 0.1, mag*-1.0, mag*0.0, mag*0.0)

    # Compile time duration record
    # Before optimization: 73.18037104606628 s
    # After unrolling optimization: 2.7542989253997803 s -  0.002396106719970703 s
    set_dir_acc_3D()
    # animation_init()
    # New structure to make residual calculation same as PD
    for f_cnt in range(500):
        animation_control(f_cnt)
        print("==================== Frame: ", frame_counter, " ====================")
        compute_xn_and_xTilde()
        Eprev = compute_energy()
        save_xPrev()
        itr_cnt = 0
        while True:
            data_rhs.fill(0)
            data_mat.fill(0)
            data_sol.fill(0)
            ti_intermediate_field.fill(0)
            ti_M_field.fill(0)
            compute_hessian_and_gradient(data_mat, data_rhs)
            if dim == 2:
                data_sol = solve_linear_system(data_mat, data_rhs, n_particles * dim,
                                               np.array(dirichlet), zero.to_numpy(),
                                               False, 0, cnt[None])
            else:
                data_sol = solve_linear_system3(data_mat, data_rhs, n_particles * dim,
                                                np.array(dirichlet), zero.to_numpy(),
                                                False, 0, cnt[None])
            alpha = 1.0
            while True:
                apply_sol(alpha, data_sol)
                step_vec = alpha * data_sol
                alpha *= 0.5
                E = compute_energy()
                if E <= Eprev:
                    break
            Eprev = E
            save_xPrev()
            if output_residual2(step_vec) < 1e-4 or itr_cnt >= 6:
                break
            itr_cnt += 1

        compute_v()
        output_aux_data(frame_counter)

        particle_pos = x.to_numpy()
        vertices_ = vertices.to_numpy()

        frame_counter += 1
        filename = f'./results/frame_{frame_counter:05d}.png'
        if dim == 2:
            draw_image(gui, filename, x.to_numpy(), mesh_offset, mesh_scale, vertices.to_numpy(), n_elements)
        else:
            update_boundary_mesh(x, boundary_pos, case_info)
            scene.input(gui)
            tina_mesh.set_face_verts(boundary_pos)
            scene.render()
            gui.set_image(scene.img)
            video_manager.write_frame(gui.get_image())
            gui.show()

    video_manager.make_video(gif=True, mp4=True)