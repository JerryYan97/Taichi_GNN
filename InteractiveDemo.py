import os
import sys
import numpy as np
import taichi as ti
from src.Utils.reader import read
from src.Utils.utils_visualization import draw_image, get_acc_field, output_3d_seq, update_boundary_mesh, get_ring_acc_field, get_ring_circle_acc_field, get_point_acc_field, get_point_acc_field_by_point
from scipy import sparse
from scipy.sparse.linalg import factorized
from src.Utils.math_tools import svd, my_svd
from numpy import linalg as LA

real = ti.f64

# Mesh load and test case selection:
test_case = 1011
case_info = read(test_case)
mesh = case_info['mesh']
dirichlet = case_info['dirichlet']
mesh_scale = case_info['mesh_scale']
mesh_offset = case_info['mesh_offset']
dim = case_info['dim']
if dim == 3:
    center = case_info['center']
    min_sphere_radius = case_info['min_sphere_radius']
# cpu 1.27 -- 1003,
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)
n_vertices = mesh.num_vertices
_, _, boundary_triangles = case_info['boundary']

# 2D and 3D scene settings:
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

    pars = tina.SimpleParticles()
    material = tina.BlinnPhong()
    scene.add_object(pars, material)


# Material settings:
# rho = 1e3
# E, nu = 5e6, 0.4  # Young's modulus and Poisson's ratio
E = 1e9
nu = 0.49
rho = 1.1e3
mu, lam = E / (2*(1+nu)), E * nu / ((1+nu)*(1-2*nu))  # Lame parameters


# Solver settings:
m_weight_positional = 1e20
dt = 1.0 / 24.0
# Backup settings:
# Bar: 10  Bunny: 50
solver_max_iteration = 1
solver_stop_residual = 0.1
# external acceleration -- counter-clock wise
ti_ex_acc = ti.Vector.field(dim, real, n_vertices)

# Taichi variables' initialization:
ti_mass = ti.field(real, n_vertices)
ti_volume = ti.field(real, n_elements)
ti_pos = ti.Vector.field(dim, real, n_vertices)
ti_pos_new = ti.Vector.field(dim, real, n_vertices)
ti_pos_init = ti.Vector.field(dim, real, n_vertices)
ti_last_pos_new = ti.Vector.field(dim, real, n_vertices)
ti_boundary_labels = ti.field(int, n_vertices)

ti_vel = ti.Vector.field(dim, real, n_vertices)
ti_vel_last = ti.Vector.field(dim, real, n_vertices)
ti_vel_del = ti.Vector.field(dim, real, n_vertices)

ti_elements = ti.Vector.field(dim + 1, int, n_elements)  # ids of three vertices of each face
ti_Dm_inv = ti.Matrix.field(dim, dim, real, n_elements)  # The inverse of the init elements -- Dm
ti_F = ti.Matrix.field(dim, dim, real, n_elements)
ti_A = ti.field(real, (n_elements * 2, dim * dim, dim * (dim + 1)))
ti_A_i = ti.field(real, shape=(dim * dim, dim * (dim + 1)))
ti_q_idx_vec = ti.field(ti.int32, (n_elements, dim * (dim + 1)))
ti_Bp = ti.Matrix.field(dim, dim, real, n_elements * 2)
ti_Sn = ti.field(real, n_vertices * dim)
# potential energy of each element(face) for linear coratated elasticity material.
ti_phi = ti.field(real, n_elements)
ti_weight_strain = ti.field(real, n_elements)
ti_weight_volume = ti.field(real, n_elements)
if dim == 3:
    ti_center = ti.Vector([center[0], center[1], center[2]])


def init():
    if dim == 2:
        ti_elements.from_numpy(mesh.faces)
    else:
        ti_elements.from_numpy(mesh.elements)

    ti_pos.from_numpy(mesh.vertices)
    ti_pos_init.from_numpy(mesh.vertices)
    ti_mass.fill(0)
    ti_volume.fill(0)
    ti_pos_new.fill(0)
    ti_last_pos_new.fill(0)
    ti_boundary_labels.fill(0)

    ti_vel.fill(0)
    ti_vel_del.fill(0)
    ti_vel_last.fill(0)

    ti_Dm_inv.fill(0)
    ti_F.fill(0)
    ti_A.fill(0)
    ti_A_i.fill(0)
    ti_q_idx_vec.fill(0)
    ti_Bp.fill(0)
    ti_Sn.fill(0)
    ti_phi.fill(0)
    ti_weight_strain.fill(0)
    ti_weight_volume.fill(0)
    ti_ex_acc.fill(0)


@ti.kernel
def set_point_acc_by_point_3D(pf_ind: ti.i32, pf_radius: ti.f64, x: ti.f32, y: ti.f32, z: ti.f32):   # set only one time
    for i in range(n_vertices):  # t_pos, pos, radius, acc
        ti_ex_acc[i] = get_point_acc_field_by_point(ti_pos_init[pf_ind], ti_pos_init[i], pf_radius, ti.Vector([x,y,z]))


# Backup Settings:
@ti.kernel
def set_exacc():
    if ti.static(dim == 2):
        for i in range(n_vertices):
            ti_ex_acc[i] = ti.Vector(get_acc_field(6, -45.0))
    else:
        for i in range(n_vertices):
            ti_ex_acc[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def set_ring_acc_3D():
    for i in range(n_vertices):
        ti_ex_acc[i] = get_ring_acc_field(0.04, 10.0, ti_center, ti_pos[i], 0.0, 3)


@ti.func
def set_ti_A_i(ele_idx, row, col, val):
    ti_A[ele_idx, row, col] = val


@ti.func
def get_ti_A_i(ele_idx):
    if ti.static(dim == 2):
        tmp_mat = ti.Matrix.rows([[ti_A[ele_idx, 0, 0], ti_A[ele_idx, 0, 1], ti_A[ele_idx, 0, 2],
                                   ti_A[ele_idx, 0, 3], ti_A[ele_idx, 0, 4], ti_A[ele_idx, 0, 5]],
                                  [ti_A[ele_idx, 1, 0], ti_A[ele_idx, 1, 1], ti_A[ele_idx, 1, 2],
                                   ti_A[ele_idx, 1, 3], ti_A[ele_idx, 1, 4], ti_A[ele_idx, 1, 5]],
                                  [ti_A[ele_idx, 2, 0], ti_A[ele_idx, 2, 1], ti_A[ele_idx, 2, 2],
                                   ti_A[ele_idx, 2, 3], ti_A[ele_idx, 2, 4], ti_A[ele_idx, 2, 5]],
                                  [ti_A[ele_idx, 3, 0], ti_A[ele_idx, 3, 1], ti_A[ele_idx, 3, 2],
                                   ti_A[ele_idx, 3, 3], ti_A[ele_idx, 3, 4], ti_A[ele_idx, 3, 5]]])
        return tmp_mat
    else:
        tmp_mat = ti.Matrix.rows([[ti_A[ele_idx, 0, 0], ti_A[ele_idx, 0, 1], ti_A[ele_idx, 0, 2], ti_A[ele_idx, 0, 3],
                                   ti_A[ele_idx, 0, 4], ti_A[ele_idx, 0, 5], ti_A[ele_idx, 0, 6], ti_A[ele_idx, 0, 7],
                                   ti_A[ele_idx, 0, 8], ti_A[ele_idx, 0, 9], ti_A[ele_idx, 0, 10],
                                   ti_A[ele_idx, 0, 11]],
                                  [ti_A[ele_idx, 1, 0], ti_A[ele_idx, 1, 1], ti_A[ele_idx, 1, 2], ti_A[ele_idx, 1, 3],
                                   ti_A[ele_idx, 1, 4], ti_A[ele_idx, 1, 5], ti_A[ele_idx, 1, 6], ti_A[ele_idx, 1, 7],
                                   ti_A[ele_idx, 1, 8], ti_A[ele_idx, 1, 9], ti_A[ele_idx, 1, 10],
                                   ti_A[ele_idx, 1, 11]],
                                  [ti_A[ele_idx, 2, 0], ti_A[ele_idx, 2, 1], ti_A[ele_idx, 2, 2], ti_A[ele_idx, 2, 3],
                                   ti_A[ele_idx, 2, 4], ti_A[ele_idx, 2, 5], ti_A[ele_idx, 2, 6], ti_A[ele_idx, 2, 7],
                                   ti_A[ele_idx, 2, 8], ti_A[ele_idx, 2, 9], ti_A[ele_idx, 2, 10],
                                   ti_A[ele_idx, 2, 11]],
                                  [ti_A[ele_idx, 3, 0], ti_A[ele_idx, 3, 1], ti_A[ele_idx, 3, 2], ti_A[ele_idx, 3, 3],
                                   ti_A[ele_idx, 3, 4], ti_A[ele_idx, 3, 5], ti_A[ele_idx, 3, 6], ti_A[ele_idx, 3, 7],
                                   ti_A[ele_idx, 3, 8], ti_A[ele_idx, 3, 9], ti_A[ele_idx, 3, 10],
                                   ti_A[ele_idx, 3, 11]],
                                  [ti_A[ele_idx, 4, 0], ti_A[ele_idx, 4, 1], ti_A[ele_idx, 4, 2], ti_A[ele_idx, 4, 3],
                                   ti_A[ele_idx, 4, 4], ti_A[ele_idx, 4, 5], ti_A[ele_idx, 4, 6], ti_A[ele_idx, 4, 7],
                                   ti_A[ele_idx, 4, 8], ti_A[ele_idx, 4, 9], ti_A[ele_idx, 4, 10],
                                   ti_A[ele_idx, 4, 11]],
                                  [ti_A[ele_idx, 5, 0], ti_A[ele_idx, 5, 1], ti_A[ele_idx, 5, 2], ti_A[ele_idx, 5, 3],
                                   ti_A[ele_idx, 5, 4], ti_A[ele_idx, 5, 5], ti_A[ele_idx, 5, 6], ti_A[ele_idx, 5, 7],
                                   ti_A[ele_idx, 5, 8], ti_A[ele_idx, 5, 9], ti_A[ele_idx, 5, 10],
                                   ti_A[ele_idx, 5, 11]],
                                  [ti_A[ele_idx, 6, 0], ti_A[ele_idx, 6, 1], ti_A[ele_idx, 6, 2], ti_A[ele_idx, 6, 3],
                                   ti_A[ele_idx, 6, 4], ti_A[ele_idx, 6, 5], ti_A[ele_idx, 6, 6], ti_A[ele_idx, 6, 7],
                                   ti_A[ele_idx, 6, 8], ti_A[ele_idx, 6, 9], ti_A[ele_idx, 6, 10],
                                   ti_A[ele_idx, 6, 11]],
                                  [ti_A[ele_idx, 7, 0], ti_A[ele_idx, 7, 1], ti_A[ele_idx, 7, 2], ti_A[ele_idx, 7, 3],
                                   ti_A[ele_idx, 7, 4], ti_A[ele_idx, 7, 5], ti_A[ele_idx, 7, 6], ti_A[ele_idx, 7, 7],
                                   ti_A[ele_idx, 7, 8], ti_A[ele_idx, 7, 9], ti_A[ele_idx, 7, 10],
                                   ti_A[ele_idx, 7, 11]],
                                  [ti_A[ele_idx, 8, 0], ti_A[ele_idx, 8, 1], ti_A[ele_idx, 8, 2], ti_A[ele_idx, 8, 3],
                                   ti_A[ele_idx, 8, 4], ti_A[ele_idx, 8, 5], ti_A[ele_idx, 8, 6], ti_A[ele_idx, 8, 7],
                                   ti_A[ele_idx, 8, 8], ti_A[ele_idx, 8, 9], ti_A[ele_idx, 8, 10],
                                   ti_A[ele_idx, 8, 11]]
                                  ])
        return tmp_mat


@ti.func
def compute_Dm(i):
    if ti.static(dim == 2):
        ia, ib, ic = ti_elements[i]
        a, b, c = ti_pos_init[ia], ti_pos_init[ib], ti_pos_init[ic]
        return ti.Matrix.cols([b - a, c - a])
    else:
        idx_a, idx_b, idx_c, idx_d = ti_elements[i]
        a, b, c, d = ti_pos_init[idx_a], ti_pos_init[idx_b], ti_pos_init[idx_c], ti_pos_init[idx_d]
        return ti.Matrix.cols([b - a, c - a, d - a])


@ti.kernel
def init_mesh_DmInv(input_dirichlet: ti.ext_arr(), input_dirichlet_num: int):
    for i in range(input_dirichlet_num):
        ti_boundary_labels[int(input_dirichlet[i])] = 1
    for i in range(n_elements):
        # Compute Dm:
        Dm_i = compute_Dm(i)
        ti_Dm_inv[i] = Dm_i.inverse()
        ti_volume[i] = ti.abs(Dm_i.determinant()) * 0.5
        ti_weight_strain[i] = mu * 2 * ti_volume[i]
        ti_weight_volume[i] = lam * dim * ti_volume[i]


@ti.func
def fill_idx_vec(ele_idx):
    if ti.static(dim == 2):
        ia, ib, ic = ti_elements[ele_idx]
        ia_x_idx, ia_y_idx = ia * 2, ia * 2 + 1
        ib_x_idx, ib_y_idx = ib * 2, ib * 2 + 1
        ic_x_idx, ic_y_idx = ic * 2, ic * 2 + 1
        ti_q_idx_vec[ele_idx, 0], ti_q_idx_vec[ele_idx, 1] = ia_x_idx, ia_y_idx
        ti_q_idx_vec[ele_idx, 2], ti_q_idx_vec[ele_idx, 3] = ib_x_idx, ib_y_idx
        ti_q_idx_vec[ele_idx, 4], ti_q_idx_vec[ele_idx, 5] = ic_x_idx, ic_y_idx
    else:
        idx_a, idx_b, idx_c, idx_d = ti_elements[ele_idx]
        idx_a_x_idx, idx_a_y_idx, idx_a_z_idx = idx_a * 3, idx_a * 3 + 1, idx_a * 3 + 2
        idx_b_x_idx, idx_b_y_idx, idx_b_z_idx = idx_b * 3, idx_b * 3 + 1, idx_b * 3 + 2
        idx_c_x_idx, idx_c_y_idx, idx_c_z_idx = idx_c * 3, idx_c * 3 + 1, idx_c * 3 + 2
        idx_d_x_idx, idx_d_y_idx, idx_d_z_idx = idx_d * 3, idx_d * 3 + 1, idx_d * 3 + 2
        ti_q_idx_vec[ele_idx, 0], ti_q_idx_vec[ele_idx, 1], ti_q_idx_vec[ele_idx, 2] = idx_a_x_idx, idx_a_y_idx, idx_a_z_idx
        ti_q_idx_vec[ele_idx, 3], ti_q_idx_vec[ele_idx, 4], ti_q_idx_vec[ele_idx, 5] = idx_b_x_idx, idx_b_y_idx, idx_b_z_idx
        ti_q_idx_vec[ele_idx, 6], ti_q_idx_vec[ele_idx, 7], ti_q_idx_vec[ele_idx, 8] = idx_c_x_idx, idx_c_y_idx, idx_c_z_idx
        ti_q_idx_vec[ele_idx, 9], ti_q_idx_vec[ele_idx, 10], ti_q_idx_vec[ele_idx, 11] = idx_d_x_idx, idx_d_y_idx, idx_d_z_idx


@ti.kernel
def precomputation(lhs_mat_row: ti.ext_arr(), lhs_mat_col: ti.ext_arr(), lhs_mat_val: ti.ext_arr()):
    dimp = dim+1
    sparse_used_idx_cnt = 0
    for e_it in range(n_elements):
        if ti.static(dim == 2):
            ia, ib, ic = ti_elements[e_it]
            ti_mass[ia] += ti_volume[e_it] / dimp * rho
            ti_mass[ib] += ti_volume[e_it] / dimp * rho
            ti_mass[ic] += ti_volume[e_it] / dimp * rho
        else:
            idx_a, idx_b, idx_c, idx_d = ti_elements[e_it]
            ti_mass[idx_a] += ti_volume[e_it] / dimp * rho
            ti_mass[idx_b] += ti_volume[e_it] / dimp * rho
            ti_mass[idx_c] += ti_volume[e_it] / dimp * rho
            ti_mass[idx_d] += ti_volume[e_it] / dimp * rho

    # Construct A_i matrix for every element / Build A for all the constraints:
    # Strain constraints and area constraints
    for i in range(n_elements):
        for t in ti.static(range(2)):
            if ti.static(dim == 2):
                # Get (Dm)^-1 for this element:
                Dm_inv_i = ti_Dm_inv[i]
                a = Dm_inv_i[0, 0]
                b = Dm_inv_i[0, 1]
                c = Dm_inv_i[1, 0]
                d = Dm_inv_i[1, 1]
                # Construct A_i:
                set_ti_A_i(t * n_elements + i, 0, 0, -a - c)
                set_ti_A_i(t * n_elements + i, 0, 2, a)
                set_ti_A_i(t * n_elements + i, 0, 4, c)
                set_ti_A_i(t * n_elements + i, 1, 0, -b - d)
                set_ti_A_i(t * n_elements + i, 1, 2, b)
                set_ti_A_i(t * n_elements + i, 1, 4, d)
                set_ti_A_i(t * n_elements + i, 2, 1, -a - c)
                set_ti_A_i(t * n_elements + i, 2, 3, a)
                set_ti_A_i(t * n_elements + i, 2, 5, c)
                set_ti_A_i(t * n_elements + i, 3, 1, -b - d)
                set_ti_A_i(t * n_elements + i, 3, 3, b)
                set_ti_A_i(t * n_elements + i, 3, 5, d)

            else:
                # Get (Dm)^-1 for this element:
                Dm_inv_i = ti_Dm_inv[i]
                e11, e12, e13 = Dm_inv_i[0, 0], Dm_inv_i[0, 1], Dm_inv_i[0, 2]
                e21, e22, e23 = Dm_inv_i[1, 0], Dm_inv_i[1, 1], Dm_inv_i[1, 2]
                e31, e32, e33 = Dm_inv_i[2, 0], Dm_inv_i[2, 1], Dm_inv_i[2, 2]
                # Construct A_i:
                set_ti_A_i(t * n_elements + i, 0, 0, -(e11 + e21 + e31))
                set_ti_A_i(t * n_elements + i, 0, 3, e11)
                set_ti_A_i(t * n_elements + i, 0, 6, e21)
                set_ti_A_i(t * n_elements + i, 0, 9, e31)
                set_ti_A_i(t * n_elements + i, 1, 0, -(e12 + e22 + e32))
                set_ti_A_i(t * n_elements + i, 1, 3, e12)
                set_ti_A_i(t * n_elements + i, 1, 6, e22)
                set_ti_A_i(t * n_elements + i, 1, 9, e32)
                set_ti_A_i(t * n_elements + i, 2, 0, -(e13 + e23 + e33))
                set_ti_A_i(t * n_elements + i, 2, 3, e13)
                set_ti_A_i(t * n_elements + i, 2, 6, e23)
                set_ti_A_i(t * n_elements + i, 2, 9, e33)

                set_ti_A_i(t * n_elements + i, 3, 1, -(e11 + e21 + e31))
                set_ti_A_i(t * n_elements + i, 3, 4, e11)
                set_ti_A_i(t * n_elements + i, 3, 7, e21)
                set_ti_A_i(t * n_elements + i, 3, 10, e31)
                set_ti_A_i(t * n_elements + i, 4, 1, -(e12 + e22 + e32))
                set_ti_A_i(t * n_elements + i, 4, 4, e12)
                set_ti_A_i(t * n_elements + i, 4, 7, e22)
                set_ti_A_i(t * n_elements + i, 4, 10, e32)
                set_ti_A_i(t * n_elements + i, 5, 1, -(e13 + e23 + e33))
                set_ti_A_i(t * n_elements + i, 5, 4, e13)
                set_ti_A_i(t * n_elements + i, 5, 7, e23)
                set_ti_A_i(t * n_elements + i, 5, 10, e33)

                set_ti_A_i(t * n_elements + i, 6, 2, -(e11 + e21 + e31))
                set_ti_A_i(t * n_elements + i, 6, 5, e11)
                set_ti_A_i(t * n_elements + i, 6, 8, e21)
                set_ti_A_i(t * n_elements + i, 6, 11, e31)
                set_ti_A_i(t * n_elements + i, 7, 2, -(e12 + e22 + e32))
                set_ti_A_i(t * n_elements + i, 7, 5, e12)
                set_ti_A_i(t * n_elements + i, 7, 8, e22)
                set_ti_A_i(t * n_elements + i, 7, 11, e32)
                set_ti_A_i(t * n_elements + i, 8, 2, -(e13 + e23 + e33))
                set_ti_A_i(t * n_elements + i, 8, 5, e13)
                set_ti_A_i(t * n_elements + i, 8, 8, e23)
                set_ti_A_i(t * n_elements + i, 8, 11, e33)

    # Sparse modification Changed:
    for ele_idx in range(n_elements):
        fill_idx_vec(ele_idx)
        ele_global_start_idx = ele_idx * dim * (dim + 1) * dim * (dim + 1)
        ele_offset_idx = 0
        for A_row_idx in range(dim * (dim + 1)):
            for A_col_idx in range(dim * (dim + 1)):
                lhs_row_idx = ti_q_idx_vec[ele_idx, A_row_idx]
                lhs_col_idx = ti_q_idx_vec[ele_idx, A_col_idx]
                weight_strain = ti_weight_strain[ele_idx]
                weight_volume = ti_weight_volume[ele_idx]
                cur_sparse_val = 0.0
                if ti.static(dim) == 2:
                    cur_sparse_val += (ti_A[ele_idx, 0, A_row_idx] * ti_A[ele_idx, 0, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 1, A_row_idx] * ti_A[ele_idx, 1, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 2, A_row_idx] * ti_A[ele_idx, 2, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 3, A_row_idx] * ti_A[ele_idx, 3, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 0, A_row_idx] * ti_A[ele_idx, 0, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 1, A_row_idx] * ti_A[ele_idx, 1, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 2, A_row_idx] * ti_A[ele_idx, 2, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 3, A_row_idx] * ti_A[ele_idx, 3, A_col_idx] * weight_volume)
                else:
                    cur_sparse_val += (ti_A[ele_idx, 0, A_row_idx] * ti_A[ele_idx, 0, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 1, A_row_idx] * ti_A[ele_idx, 1, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 2, A_row_idx] * ti_A[ele_idx, 2, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 3, A_row_idx] * ti_A[ele_idx, 3, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 4, A_row_idx] * ti_A[ele_idx, 4, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 5, A_row_idx] * ti_A[ele_idx, 5, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 6, A_row_idx] * ti_A[ele_idx, 6, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 7, A_row_idx] * ti_A[ele_idx, 7, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 8, A_row_idx] * ti_A[ele_idx, 8, A_col_idx] * weight_strain)
                    cur_sparse_val += (ti_A[ele_idx, 0, A_row_idx] * ti_A[ele_idx, 0, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 1, A_row_idx] * ti_A[ele_idx, 1, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 2, A_row_idx] * ti_A[ele_idx, 2, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 3, A_row_idx] * ti_A[ele_idx, 3, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 4, A_row_idx] * ti_A[ele_idx, 4, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 5, A_row_idx] * ti_A[ele_idx, 5, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 6, A_row_idx] * ti_A[ele_idx, 6, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 7, A_row_idx] * ti_A[ele_idx, 7, A_col_idx] * weight_volume)
                    cur_sparse_val += (ti_A[ele_idx, 8, A_row_idx] * ti_A[ele_idx, 8, A_col_idx] * weight_volume)
                # lhs_matrix_np[lhs_row_idx, lhs_col_idx] += cur_sparse_val
                cur_idx = ele_global_start_idx + ele_offset_idx
                lhs_mat_row[cur_idx] = lhs_row_idx
                lhs_mat_col[cur_idx] = lhs_col_idx
                lhs_mat_val[cur_idx] = cur_sparse_val
                ele_offset_idx += 1
    sparse_used_idx_cnt += n_elements * dim * (dim + 1) * dim * (dim + 1)

    # Add positional constraints and mass terms to the lhs matrix
    for i in range(n_vertices):
        global_start_idx = sparse_used_idx_cnt + i * dim
        local_offset_idx = 0
        for d in range(dim):
            cur_sparse_val = 0.0
            cur_sparse_val += (ti_mass[i] / (dt * dt))
            if ti_boundary_labels[i] == 1:
                cur_sparse_val += m_weight_positional
            lhs_row_idx, lhs_col_idx = i * dim + d, i * dim + d
            cur_idx = global_start_idx + local_offset_idx
            lhs_mat_row[cur_idx] = lhs_row_idx
            lhs_mat_col[cur_idx] = lhs_col_idx
            lhs_mat_val[cur_idx] = cur_sparse_val
            local_offset_idx += 1


@ti.func
def compute_Fi(i):
    if ti.static(dim == 2):
        ia, ib, ic = ti_elements[i]
        a, b, c = ti_pos_new[ia], ti_pos_new[ib], ti_pos_new[ic]
        D_i = ti.Matrix.cols([b - a, c - a])
        return ti.cast(D_i @ ti_Dm_inv[i], real)
    else:
        idx_a, idx_b, idx_c, idx_d = ti_elements[i]
        a, b, c, d = ti_pos_new[idx_a], ti_pos_new[idx_b], ti_pos_new[idx_c], ti_pos_new[idx_d]
        D_i = ti.Matrix.cols([b - a, c - a, d - a])
        return ti.cast(D_i @ ti_Dm_inv[i], real)


@ti.func
def compute_volume_constraint(sigma):
    if ti.static(dim == 2):
        sigma_star_11_k, sigma_star_22_k = 1.0, 1.0
        for itr in ti.static(range(10)):
            first_term = (sigma_star_11_k * sigma_star_22_k - sigma[0, 0] * sigma_star_22_k - sigma[
                1, 1] * sigma_star_11_k + 1.0) / (sigma_star_11_k ** 2 + sigma_star_22_k ** 2)
            D1_kplus1 = first_term * sigma_star_22_k
            D2_kplus1 = first_term * sigma_star_11_k
            sigma_star_11_k = D1_kplus1 + sigma[0, 0]
            sigma_star_22_k = D2_kplus1 + sigma[1, 1]
        return ti.Matrix.rows([[sigma_star_11_k, 0.0], [0.0, sigma_star_22_k]])
    else:
        tol = 0.00001
        max_it = 1
        s1, s2, s3 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
        x = 1.0 - s1
        y = 1.0 - s2
        z = 1.0 - s3
        for itr in range(max_it):
            a, b, c = x + s1, y + s2, z + s3
            f = a * b * c - 1.0
            g1, g2, g3 = b * c, a * c, a * b
            bot = g1 * g1 + g2 * g2 + g3 * g3
            if ti.abs(bot) < tol:
                break
            top = x * g1 + y * g2 + z * g3 - f
            div = top / bot
            x0, y0, z0 = x, y, z
            x = div * g1
            y = div * g2
            z = div * g3
            dx = x - x0
            dy = y - y0
            dz = z - z0
            if dx * dx + dy * dy + dz * dz < tol * tol:
                break
        return ti.Matrix.rows([[x + s1, 0.0, 0.0], [0.0, y + s2, 0.0], [0.0, 0.0, z + s3]])


# NOTE: This function doesn't build all constraints
# It just builds strain constraints and area/volume constraints
@ti.kernel
def local_solve_build_bp_for_all_constraints():
    for i in range(n_elements):
        # Construct strain constraints:
        # Construct Current F_i:
        F_i = compute_Fi(i)
        ti_F[i] = F_i
        # Use current F_i construct current 'B * p' or Ri
        # U, sigma, V = svd(F_i)
        U, sigma, V = my_svd(F_i)
        # U, sigma, V = ti.svd(F_i)
        ti_Bp[i] = U @ V.transpose()

        # Construct volume preservation constraints:
        # My test:
        PP = compute_volume_constraint(sigma)
        ti_Bp[n_elements + i] = U @ PP @ V.transpose()

    # Calculate Phi for all the elements:
    for i in range(n_elements):
        Bp_i_strain = ti_Bp[i]
        Bp_i_volume = ti_Bp[n_elements + i]
        F_i = ti_F[i]
        energy1 = mu * ti_volume[i] * ((F_i - Bp_i_strain).norm() ** 2)
        energy2 = 0.5 * lam * ti_volume[i] * ((F_i - Bp_i_volume).trace() ** 2)
        ti_phi[i] = energy1 + energy2


@ti.kernel
def build_sn():
    for v_id in range(n_vertices):  # number of vertices
        Sn_idx1 = v_id*dim
        Sn_idx2 = v_id*dim+1
        pos_i = ti_pos[v_id]
        vel_i = ti_vel[v_id]
        ti_Sn[Sn_idx1] = pos_i[0] + dt * vel_i[0]+(dt**2)*(ti_ex_acc[v_id][0])
        ti_Sn[Sn_idx2] = pos_i[1] + dt * vel_i[1]+(dt**2)*(ti_ex_acc[v_id][1])
        if ti.static(dim == 3):
            Sn_idx3 = v_id * dim + 2
            ti_Sn[Sn_idx3] = pos_i[2]+dt*vel_i[2]+(dt**2)*(ti_ex_acc[v_id][2])


@ti.func
def Build_Bp_i_vec(idx):
    Bp_i = ti_Bp[idx]
    if ti.static(dim == 2):
        Bp_i_vec = ti.Vector([Bp_i[0, 0], Bp_i[0, 1],
                              Bp_i[1, 0], Bp_i[1, 1]])
        return Bp_i_vec
    else:
        Bp_i_vec = ti.Vector([Bp_i[0, 0], Bp_i[0, 1], Bp_i[0, 2],
                              Bp_i[1, 0], Bp_i[1, 1], Bp_i[1, 2],
                              Bp_i[2, 0], Bp_i[2, 1], Bp_i[2, 2]])
        return Bp_i_vec


@ti.kernel
def build_rhs(rhs: ti.ext_arr()):  # dp_pos: ti.ext_arr(), dp_vel: ti.ext_arr()):
    one_over_dt2 = 1.0 / (dt ** 2)
    # Construct the first part of the rhs
    for i in range(n_vertices * dim):
        rhs[i] = one_over_dt2 * ti_mass[i // dim] * ti_Sn[i]
    # Add strain and volume/area constraints to the rhs
    for t in ti.static(range(2)):
        for ele_idx in range(n_elements):
            Bp_i_vec = Build_Bp_i_vec(t * n_elements+ele_idx)
            A_i = get_ti_A_i(ele_idx)
            AT_Bp = A_i.transpose() @ Bp_i_vec
            weight = 0.0
            if t == 0:
                weight = ti_weight_strain[ele_idx]
            else:
                weight = ti_weight_volume[ele_idx]
            AT_Bp *= weight  # m_weight_strain

            if ti.static(dim == 2):
                ia, ib, ic = ti_elements[ele_idx]

                # Add AT_Bp back to rhs
                q_ia_x_idx = ia * 2
                q_ia_y_idx = q_ia_x_idx + 1
                rhs[q_ia_x_idx] += AT_Bp[0]
                rhs[q_ia_y_idx] += AT_Bp[1]

                q_ib_x_idx = ib * 2
                q_ib_y_idx = q_ib_x_idx + 1
                rhs[q_ib_x_idx] += AT_Bp[2]
                rhs[q_ib_y_idx] += AT_Bp[3]

                q_ic_x_idx = ic * 2
                q_ic_y_idx = q_ic_x_idx + 1
                rhs[q_ic_x_idx] += AT_Bp[4]
                rhs[q_ic_y_idx] += AT_Bp[5]
            else:
                idx_a, idx_b, idx_c, idx_d = ti_elements[ele_idx]
                q_ia_x_idx = idx_a * 3
                q_ia_y_idx = q_ia_x_idx + 1
                q_ia_z_idx = q_ia_x_idx + 2
                rhs[q_ia_x_idx] += AT_Bp[0]
                rhs[q_ia_y_idx] += AT_Bp[1]
                rhs[q_ia_z_idx] += AT_Bp[2]

                q_ib_x_idx = idx_b * 3
                q_ib_y_idx = q_ib_x_idx + 1
                q_ib_z_idx = q_ib_x_idx + 2
                rhs[q_ib_x_idx] += AT_Bp[3]
                rhs[q_ib_y_idx] += AT_Bp[4]
                rhs[q_ib_z_idx] += AT_Bp[5]

                q_ic_x_idx = idx_c * 3
                q_ic_y_idx = q_ic_x_idx + 1
                q_ic_z_idx = q_ic_x_idx + 2
                rhs[q_ic_x_idx] += AT_Bp[6]
                rhs[q_ic_y_idx] += AT_Bp[7]
                rhs[q_ic_z_idx] += AT_Bp[8]

                q_id_x_idx = idx_d * 3
                q_id_y_idx = q_id_x_idx + 1
                q_id_z_idx = q_id_x_idx + 2
                rhs[q_id_x_idx] += AT_Bp[9]
                rhs[q_id_y_idx] += AT_Bp[10]
                rhs[q_id_z_idx] += AT_Bp[11]

    # Add positional constraints Bp to the rhs
    for i in range(n_vertices):
        if ti_boundary_labels[i] == 1:
            pos_init_i = ti_pos_init[i]
            q_i_x_idx = i * dim
            q_i_y_idx = i * dim + 1
            rhs[q_i_x_idx] += (pos_init_i[0] * m_weight_positional)
            rhs[q_i_y_idx] += (pos_init_i[1] * m_weight_positional)
            if ti.static(dim == 3):
                q_i_z_idx = i * dim + 2
                rhs[q_i_z_idx] += (pos_init_i[2] * m_weight_positional)


@ti.kernel
def update_velocity_pos():
    for i in range(n_vertices):
        ti_vel[i] = (ti_pos_new[i] - ti_pos[i]) / dt    # vel
        ti_vel_del[i] = ti_vel[i] - ti_vel_last[i]
        ti_pos[i] = ti_pos_new[i]   # pos


@ti.kernel
def warm_up():
    for pos_idx in range(n_vertices):
        sn_idx1, sn_idx2 = pos_idx * dim, pos_idx * dim + 1
        ti_pos_new[pos_idx][0] = ti_Sn[sn_idx1]
        ti_pos_new[pos_idx][1] = ti_Sn[sn_idx2]
        if ti.static(dim == 3):
            sn_idx3 = pos_idx * dim + 2
            ti_pos_new[pos_idx][2] = ti_Sn[sn_idx3]


@ti.kernel
def update_pos_new_from_numpy(sol: ti.ext_arr()):
    for pos_idx in range(n_vertices):
        sol_idx1, sol_idx2 = pos_idx * dim, pos_idx * dim + 1
        ti_pos_new[pos_idx][0] = sol[sol_idx1]
        ti_pos_new[pos_idx][1] = sol[sol_idx2]
        if ti.static(dim == 3):
            sol_idx3 = pos_idx * dim + 2
            ti_pos_new[pos_idx][2] = sol[sol_idx3]


@ti.kernel
def check_residual() -> ti.f32:
    residual = 0.0
    for i in range(n_vertices):
        residual += (ti_last_pos_new[i] - ti_pos_new[i]).norm()
        ti_last_pos_new[i] = ti_pos_new[i]
    # print("residual:", residual)
    return residual


@ti.kernel
def check_acceleration_status() -> ti.i32:
    times = 0
    for i in range(n_vertices):
        if (ti_vel_del[i]/dt).norm() < 0.001:
            times = times + 1
        ti_vel_last[i] = ti_vel[i]
    return times


@ti.kernel
def pos_init():
    for i in range(n_vertices):
        ti_pos[i] = ti.Vector([float(ti.random()) * 0.05, float(ti.random()) * 0.05, float(ti.random()) * 0.05])


@ti.kernel
def raySphereIntersect(ray_dir: ti.ext_arr(), ray_origin: ti.ext_arr(),
                       s_center: ti.ext_arr(), dist_array: ti.ext_arr(), vert_num: ti.i32):
    radius = 0.1
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
                dist_array[i] = l2_len


@ti.kernel
def setAcc(idx: ti.i32, acc_val: ti.ext_arr()):
    ti_ex_acc[idx][0] = acc_val[0]
    ti_ex_acc[idx][1] = acc_val[1]
    ti_ex_acc[idx][2] = acc_val[2]


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for root, dirs, files in os.walk("../results/"):
        for name in files:
            os.remove(os.path.join(root, name))

    video_manager = ti.VideoManager(output_dir=os.getcwd() + '../results/', framerate=24, automatic_build=False)
    frame_counter = 0
    rhs_np = np.zeros(n_vertices * dim, dtype=np.float64)
    lhs_mat_val = np.zeros(shape=(n_elements * dim ** 2 * (dim+1) ** 2 + n_vertices * dim,), dtype=np.float64)
    lhs_mat_row = np.zeros(shape=(n_elements * dim ** 2 * (dim+1) ** 2 + n_vertices * dim,), dtype=np.float64)
    lhs_mat_col = np.zeros(shape=(n_elements * dim ** 2 * (dim+1) ** 2 + n_vertices * dim,), dtype=np.float64)

    init()

    # One direction acc field
    # set_exacc()
    ti_ex_acc.fill(0.0)
    init_mesh_DmInv(dirichlet, len(dirichlet))
    precomputation(lhs_mat_row, lhs_mat_col, lhs_mat_val)
    s_lhs_matrix_np = sparse.csr_matrix((lhs_mat_val, (lhs_mat_row, lhs_mat_col)),
                                        shape=(n_vertices * dim, n_vertices * dim),
                                        dtype=np.float64)
    pre_fact_lhs_solve = factorized(s_lhs_matrix_np)

    if dim == 2:
        gui = ti.GUI('PN Standalone', background_color=0xf7f7f7)
        filename = f'./results/frame_rest.png'
        draw_image(gui, filename, ti_pos.to_numpy(), mesh_offset, mesh_scale, ti_elements.to_numpy(), n_elements)
    else:
        gui = ti.GUI('PD standalone 3D')
        # model.set_transform(case_info['transformation_mat'])
        update_boundary_mesh(ti_pos, boundary_pos, case_info)
        scene.input(gui)
        tina_mesh.set_face_verts(boundary_pos)
        scene.render()
        gui.set_image(scene.img)
        gui.show()

    frame_counter = 0
    sim_t = 0.0
    plot_array = []
    selected_vert_idx = -1

    while True:
        build_sn()
        # Warm up:
        warm_up()
        # print("Frame ", frame_counter)
        while True:
            local_solve_build_bp_for_all_constraints()
            build_rhs(rhs_np)

            pos_new_np = pre_fact_lhs_solve(rhs_np)
            update_pos_new_from_numpy(pos_new_np)

            residual = check_residual()
            if residual < solver_stop_residual:
                break

        # Update velocity and positions
        update_velocity_pos()

        frame_counter += 1
        filename = f'./results/frame_{frame_counter:05d}.png'  # NOTE: It needs to be moved to another place.
        if dim == 2:
            draw_image(gui, filename, ti_pos.to_numpy(), mesh_offset, mesh_scale, ti_elements.to_numpy(), n_elements)
        else:
            if selected_vert_idx != -1:
                pos_np = ti_pos.to_numpy()
                selected_vert = np.zeros((1, 3), dtype=float)
                selected_vert[0, 0] = pos_np[selected_vert_idx, 0]
                selected_vert[0, 1] = pos_np[selected_vert_idx, 1]
                selected_vert[0, 2] = pos_np[selected_vert_idx, 2]
                pars.set_particles(selected_vert)
                particles_color = np.full((1, 3), 1.0, dtype=float)
                particles_color[0, 1] -= 1.0
                particles_color[0, 2] -= 1.0
                pars.set_particle_colors(particles_color)

            update_boundary_mesh(ti_pos, boundary_pos, case_info)
            scene.input(gui)
            tina_mesh.set_face_verts(boundary_pos)
            scene.render()
            gui.set_image(scene.img)
            video_manager.write_frame(gui.get_image())
            gui.show()

            cam_pos = scene.control.center + scene.control.back
            cam_up = scene.control.up / LA.norm(scene.control.up)
            cam_front = -scene.control.back / LA.norm(scene.control.back)
            cam_right = np.cross(cam_front, cam_up)
            cam_right /= LA.norm(cam_right)

            ti_ex_acc.fill(0.0)
            if gui.is_pressed(ti.GUI.LMB):
                # Select
                mouse_x, mouse_y = gui.get_cursor_pos()
                relative_x = mouse_x - 0.5
                relative_y = mouse_y - 0.5
                sp_x = mouse_x * 2.0 - 1.0
                sp_y = mouse_y * 2.0 - 1.0
                sp_pos = np.array([sp_x, sp_y, 0.0, 1.0])
                V2W_np = scene.engine.V2W.to_numpy()
                wp_pos = V2W_np @ (sp_pos * 0.2)
                ray_dir = wp_pos[0:3] - cam_pos
                ray_dir /= LA.norm(ray_dir)
                dist_arr = np.ones(mesh.num_vertices, dtype=float) * 100000.0

                pos_np = ti_pos.to_numpy()

                raySphereIntersect(ray_dir, cam_pos, pos_np, dist_arr, mesh.num_vertices)
                min_idx = np.argmin(dist_arr)
                min_val = np.amin(dist_arr)

                if min_val != 100000.0:
                    selected_vert = np.zeros((1, 3), dtype=float)
                    selected_vert[0, 0] = pos_np[min_idx, 0]
                    selected_vert[0, 1] = pos_np[min_idx, 1]
                    selected_vert[0, 2] = pos_np[min_idx, 2]
                    pars.set_particles(selected_vert)

                    particles_color = np.full((1, 3), 1.0, dtype=float)
                    particles_color[0, 1] -= 1.0
                    particles_color[0, 2] -= 1.0
                    pars.set_particle_colors(particles_color)
                    selected_vert_idx = min_idx
                else:
                    selected_vert_idx = -1
            elif gui.is_pressed(ti.GUI.RMB):
                if selected_vert_idx != -1:
                    mouse_x, mouse_y = gui.get_cursor_pos()
                    relative_x = mouse_x - 0.5
                    relative_y = mouse_y - 0.5
                    acc1 = relative_x * cam_right
                    acc2 = relative_y * cam_up
                    acc_val = (acc1 + acc2) * 10.0
                    setAcc(selected_vert_idx.item(), acc_val)

        # if check_acceleration_status() > 800 and frame_counter > 500:
        if frame_counter > 2000:
            break

    video_manager.make_video(gif=True, mp4=True)