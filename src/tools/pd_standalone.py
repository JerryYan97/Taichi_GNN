import taichi as ti
import numpy as np
import sys, os, time
import csv
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.reader import read
from Utils.utils_visualization import draw_image

real = ti.f64

ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

# Mesh load and test case selection:
test_case = 1
case_info = read(int(test_case))
mesh = case_info['mesh']
dirichlet = case_info['dirichlet']
mesh_scale = case_info['mesh_scale']
mesh_offset = case_info['mesh_offset']
dim = case_info['dim']

n_elements = mesh.num_faces
# if dim == 2:
#     n_elements = mesh.num_faces
# else:
#     n_elements = mesh.num_elements

n_vertices = mesh.num_vertices

# Material settings:
rho = 100
E, nu = 1e4, 0.4  # Young's modulus and Poisson's ratio
mu, lam = E / (2*(1+nu)), E * nu / ((1+nu)*(1-2*nu))  # Lame parameters

# Solver settings:
# m_weight_strain = mu * 2 * volume
# m_weight_volume = lam * dim * volume
# m_weight_strain = 0.0
# m_weight_volume = 0.0
m_weight_positional = 10000.0
dt = 0.01
solver_max_iteration = 10
solver_stop_residual = 0.0001
# external force -- counter-clock wise
exf_angle = -45.0
exf_mag = 6
ti_ex_force = ti.Vector.field(dim, real, 1)


# Taichi variables' initialization:
ti_mass = ti.field(real, n_vertices)
ti_volume = ti.field(real, n_elements)
ti_pos = ti.Vector.field(2, real, n_vertices)
ti_pos_new = ti.Vector.field(2, real, n_elements)
ti_pos_init = ti.Vector.field(2, real, n_vertices)
ti_last_pos_new = ti.Vector.field(2, real, n_elements)
ti_boundary_labels = ti.field(int, n_vertices)
ti_vel = ti.Vector.field(2, real, n_vertices)
ti_f2v = ti.Vector.field(3, int, n_elements)  # ids of three vertices of each face
ti_B = ti.Matrix.field(2, 2, real, n_elements)  # The inverse of the init elements -- Dm
ti_F = ti.Matrix.field(2, 2, real, n_elements)
ti_A = ti.Matrix.field(4, 6, real, n_elements * 2)
ti_Bp = ti.Matrix.field(2, 2, real, n_elements * 2)
ti_rhs_np = np.zeros(n_vertices * 2, dtype=np.float64)
ti_Sn = ti.field(real, n_vertices * 2)
ti_lhs_matrix = ti.field(real, shape=(n_vertices * 2, n_vertices * 2))
# potential energy of each element(face) for linear coratated elasticity material.
ti_phi = ti.field(real, n_elements)
ti_weight_strain = ti.field(real, n_elements)
ti_weight_volume = ti.field(real, n_elements)


def set_exforce(angle, mag):
    x = float(mag) * ti.cos(3.1415926 / 180.0 * angle)
    y = float(mag) * ti.sin(3.1415926 / 180.0 * angle)
    ti_ex_force[0] = ti.Vector([x, y])


@ti.kernel
def init_mesh_B(input_dirichlet: ti.ext_arr(), input_dirichlet_num: int):
    for i in range(n_vertices):
        ti_vel[i] = ti.Vector([0, 0])
    for i in range(input_dirichlet_num):
        ti_boundary_labels[int(input_dirichlet[i])] = 1
    for i in range(n_elements):
        ia, ib, ic = ti_f2v[i]
        a, b, c = ti_pos_init[ia], ti_pos_init[ib], ti_pos_init[ic]
        B_i_inv = ti.Matrix.cols([b - a, c - a])  # rest B
        ti_B[i] = B_i_inv.inverse()  # rest of B inverse
        ti_volume[i] = ti.abs(B_i_inv.determinant()) * 0.5
        ti_weight_strain[i] = mu * 2 * ti_volume[i]
        ti_weight_volume[i] = lam * dim * ti_volume[i]


@ti.kernel
def precomputation():
    dimp = dim+1
    for e_it in range(n_elements):
        ia, ib, ic = ti_f2v[e_it]
        ti_mass[ia] += ti_volume[e_it]/dimp * rho
        ti_mass[ib] += ti_volume[e_it]/dimp * rho
        ti_mass[ic] += ti_volume[e_it]/dimp * rho

    # Construct A_i matrix for every element / Build A for all the constraints:
    # Strain constraints and area constraints
    for t in ti.static(range(2)):
        for i in range(n_elements):
            # Get (Dm)^-1 for this element:
            Dm_inv_i = ti_B[i]
            a = Dm_inv_i[0, 0]
            b = Dm_inv_i[0, 1]
            c = Dm_inv_i[1, 0]
            d = Dm_inv_i[1, 1]
            # Construct A_i:
            ti_A[t*n_elements+i][0, 0] = -a-c
            ti_A[t*n_elements+i][0, 2] = a
            ti_A[t*n_elements+i][0, 4] = c
            ti_A[t*n_elements+i][1, 0] = -b-d
            ti_A[t*n_elements+i][1, 2] = b
            ti_A[t*n_elements+i][1, 4] = d
            ti_A[t*n_elements+i][2, 1] = -a-c
            ti_A[t*n_elements+i][2, 3] = a
            ti_A[t*n_elements+i][2, 5] = c
            ti_A[t*n_elements+i][3, 1] = -b-d
            ti_A[t*n_elements+i][3, 3] = b
            ti_A[t*n_elements+i][3, 5] = d

    # Construct lhs matrix without constraints
    for i in range(n_vertices):
        for d in ti.static(range(2)):
            ti_lhs_matrix[i * dim + d, i * dim + d] += ti_mass[i] / (dt * dt)

    # Add strain and area/volume constraints to the lhs matrix
    for t in ti.static(range(2)):
        for ele_idx in range(n_elements):
            A_i = ti_A[t*n_elements+ele_idx]
            ia, ib, ic = ti_f2v[ele_idx]
            ia_x_idx, ia_y_idx = ia*2, ia*2+1
            ib_x_idx, ib_y_idx = ib*2, ib*2+1
            ic_x_idx, ic_y_idx = ic*2, ic*2+1
            q_idx_vec = ti.Vector([ia_x_idx, ia_y_idx, ib_x_idx, ib_y_idx, ic_x_idx, ic_y_idx])
            # AT_A = A_i.transpose() @ A_i
            for A_row_idx in ti.static(range(6)):
                for A_col_idx in ti.static(range(6)):
                    lhs_row_idx = q_idx_vec[A_row_idx]
                    lhs_col_idx = q_idx_vec[A_col_idx]
                    for idx in ti.static(range(4)):
                        weight = 0.0
                        if t == 0:
                            weight = ti_weight_strain[ele_idx]
                        else:
                            weight = ti_weight_volume[ele_idx]
                        ti_lhs_matrix[lhs_row_idx, lhs_col_idx] += (A_i[idx, A_row_idx] * A_i[idx, A_col_idx] * weight)

    # Add positional constraints to the lhs matrix
    for i in range(n_vertices):
        if ti_boundary_labels[i] == 1:
            q_i_x_idx = i * 2
            q_i_y_idx = i * 2 + 1
            ti_lhs_matrix[q_i_x_idx, q_i_x_idx] += m_weight_positional  # This is the weight of positional constraints
            ti_lhs_matrix[q_i_y_idx, q_i_y_idx] += m_weight_positional


# NOTE: This function doesn't build all constraints
# It just builds strain constraints and area/volume constraints
@ti.kernel
def local_solve_build_bp_for_all_constraints():
    for i in range(n_elements):
        # Construct strain constraints:
        # Construct Current F_i:
        ia, ib, ic = ti_f2v[i]
        a, b, c = ti_pos_new[ia], ti_pos_new[ib], ti_pos_new[ic]
        D_i = ti.Matrix.cols([b - a, c - a])
        F_i = ti.cast(D_i @ ti_B[i], real)
        ti_F[i] = F_i
        # Use current F_i construct current 'B * p' or Ri
        U, sigma, V = ti.svd(F_i, real)
        ti_Bp[i] = U @ V.transpose()

        # Construct volume preservation constraints:
        x, y, max_it, tol = 10.0, 10.0, 80, 1e-6
        for t in range(max_it):
            aa, bb = x + sigma[0, 0], y + sigma[1, 1]
            f = aa * bb - 1
            g1, g2 = bb, aa
            bot = g1 * g1 + g2 * g2
            if abs(bot) < tol:
                break
            top = x * g1 + y * g2 - f
            div = top / bot
            x0, y0 = x, y
            x = div * g1
            y = div * g2
            _dx, _dy = x - x0, y - y0
            if _dx * _dx + _dy * _dy < tol * tol:
                break
        PP = ti.Matrix.rows([[x + sigma[0, 0], 0.0], [0.0, sigma[1, 1] + y]])
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
    for vert_idx in range(n_vertices):  # number of vertices
        Sn_idx1 = vert_idx*2  # m_sn
        Sn_idx2 = vert_idx*2+1
        pos_i = ti_pos[vert_idx]  # pos = m_x
        vel_i = ti_vel[vert_idx]
        ti_Sn[Sn_idx1] = pos_i[0] + dt * vel_i[0] + (dt ** 2) * ti_ex_force[0][0] / ti_mass[vert_idx]  # x-direction;
        ti_Sn[Sn_idx2] = pos_i[1] + dt * vel_i[1] + (dt ** 2) * ti_ex_force[0][1] / ti_mass[vert_idx]  # y-direction;


@ti.kernel
def build_rhs(rhs: ti.ext_arr()):
    one_over_dt2 = 1.0 / (dt ** 2)
    # Construct the first part of the rhs
    for i in range(n_vertices * 2):
        if i % 2 == 0:
            rhs[i] = one_over_dt2 * ti_mass[i/2] * ti_Sn[i]
        else:
            rhs[i] = one_over_dt2 * ti_mass[i/2] * ti_Sn[i]
    # Add strain and volume/area constraints to the rhs
    for t in ti.static(range(2)):
        for ele_idx in range(n_elements):
            ia, ib, ic = ti_f2v[ele_idx]
            Bp_i = ti_Bp[t*n_elements+ele_idx]  # It is a 2x2 matrix now. We want it be a 4x1 vector.
            Bp_i_vec = ti.Vector([Bp_i[0, 0], Bp_i[0, 1], Bp_i[1, 0], Bp_i[1, 1]])
            A_i = ti_A[ele_idx]
            AT_Bp = A_i.transpose() @ Bp_i_vec  # AT_Bp is a 6x1 vector now.
            weight = 0.0
            if t == 0:
                weight = ti_weight_strain[ele_idx]
            else:
                weight = ti_weight_volume[ele_idx]
            AT_Bp *= weight  # m_weight_strain

            # Add AT_Bp back to rhs
            q_ia_x_idx = ia*2
            q_ia_y_idx = q_ia_x_idx+1
            rhs[q_ia_x_idx] += AT_Bp[0]
            rhs[q_ia_y_idx] += AT_Bp[1]

            q_ib_x_idx = ib*2
            q_ib_y_idx = q_ib_x_idx+1
            rhs[q_ib_x_idx] += AT_Bp[2]
            rhs[q_ib_y_idx] += AT_Bp[3]

            q_ic_x_idx = ic*2
            q_ic_y_idx = q_ic_x_idx+1
            rhs[q_ic_x_idx] += AT_Bp[4]
            rhs[q_ic_y_idx] += AT_Bp[5]
    # Add positional constraints Bp to the rhs
    for i in range(n_vertices):
        if ti_boundary_labels[i] == 1:
            pos_init_i = ti_pos_init[i]
            q_i_x_idx = i * 2
            q_i_y_idx = i * 2 + 1
            rhs[q_i_x_idx] += (pos_init_i[0] * m_weight_positional)
            rhs[q_i_y_idx] += (pos_init_i[1] * m_weight_positional)


@ti.kernel
def update_velocity_pos():
    for i in range(n_vertices):
        ti_vel[i] = (ti_pos_new[i] - ti_pos[i]) / dt
        ti_pos[i] = ti_pos_new[i]


@ti.kernel
def warm_up():
    for pos_idx in range(n_vertices):
        sn_idx1, sn_idx2 = pos_idx * 2, pos_idx * 2 + 1
        ti_pos_new[pos_idx][0] = ti_Sn[sn_idx1]
        ti_pos_new[pos_idx][1] = ti_Sn[sn_idx2]


@ti.kernel
def update_pos_new_from_numpy(sol: ti.ext_arr()):
    for pos_idx in range(n_vertices):
        sol_idx1, sol_idx2 = pos_idx*2, pos_idx*2+1
        ti_pos_new[pos_idx][0] = sol[sol_idx1]
        ti_pos_new[pos_idx][1] = sol[sol_idx2]


@ti.kernel
def check_residual() -> ti.f32:
    residual = 0.0
    for i in range(n_vertices):
        residual += (ti_last_pos_new[i] - ti_pos_new[i]).norm()
        ti_last_pos_new[i] = ti_pos_new[i]
    # print("residual:", residual)
    return residual


@ti.kernel
def compute_T1_energy() -> real:
    T1 = 0.0
    for i in range(n_vertices):
        sn_idx1, sn_idx2 = i * 2, i * 2 + 1
        sn_i = ti.Vector([ti_Sn[sn_idx1], ti_Sn[sn_idx2]])
        temp_diff = (ti_pos_new[i] - sn_i) * ti.sqrt(ti_mass[i])
        T1 += (temp_diff[0]**2 + temp_diff[1]**2)
    return T1 / (2.0 * dt**2)


@ti.kernel
def global_compute_T2_energy() -> real:
    T2_global_energy = ti.cast(0.0, real)
    # Calculate the energy contributed by strain and volume/area constraints
    for i in range(n_elements):
        # Construct Current F_i
        ia, ib, ic = ti_f2v[i]
        a, b, c = ti_pos_new[ia], ti_pos_new[ib], ti_pos_new[ic]
        D_i = ti.Matrix.cols([b - a, c - a])
        F_i = ti.cast(D_i @ ti_B[i], real)
        # Get current Bp
        Bp_i_strain = ti_Bp[i]
        Bp_i_volume = ti_Bp[n_elements + i]
        energy1 = ti_weight_strain[i] * ((F_i - Bp_i_strain).norm() ** 2) / ti.cast(2.0, real)
        energy2 = ti_weight_volume[i] * ((F_i - Bp_i_volume).norm() ** 2) / ti.cast(2.0, real)
        T2_global_energy += (energy1 + energy2)
    # Calculate the energy contributed by positional constraints
    for i in range(n_vertices):
        if ti_boundary_labels[i] == 1:
            pos_init_i = ti_pos_init[i]
            pos_curr_i = ti_pos_new[i]
            energy3 = m_weight_positional * ((pos_curr_i - pos_init_i).norm() ** 2) / ti.cast(2.0, real)
            T2_global_energy += energy3
    return T2_global_energy


@ti.kernel
def local_compute_T2_energy() -> real:
    # Calculate T2 energy
    local_T2_energy = ti.cast(0.0, real)
    # Calculate the energy contributed by strain and volume/area constraints
    for e_it in range(n_elements):
        Bp_i_strain = ti_Bp[e_it]
        Bp_i_volume = ti_Bp[e_it + n_elements]
        F_i = ti_F[e_it]
        energy1 = ti_weight_strain[e_it] * ((F_i - Bp_i_strain).norm() ** 2) / ti.cast(2.0, real)
        energy2 = ti_weight_volume[e_it] * ((F_i - Bp_i_volume).norm() ** 2) / ti.cast(2.0, real)
        local_T2_energy += (energy1 + energy2)
    # Calculate the energy contributed by positional constraints
    for i in range(n_vertices):
        if ti_boundary_labels[i] == 1:
            pos_init_i = ti_pos_init[i]
            pos_curr_i = ti_pos_new[i]
            energy3 = m_weight_positional * ((pos_curr_i - pos_init_i).norm() ** 2) / ti.cast(2.0, real)
            local_T2_energy += energy3
    return local_T2_energy


def compute_global_step_energy():
    # Calculate global T2 energy
    global_T2_energy = global_compute_T2_energy()
    # Calculate global T1 energy
    global_T1_energy = compute_T1_energy()
    return (global_T1_energy + global_T2_energy)


def compute_local_step_energy():
    local_T2_energy = local_compute_T2_energy()
    # Calculate T1 energy
    local_T1_energy = compute_T1_energy()
    return (local_T1_energy + local_T2_energy)


if __name__ == "__main__":
    frame_counter = 0
    rhs_np = np.zeros(n_vertices * 2, dtype=np.float64)
    ti_f2v.from_numpy(mesh.faces)
    ti_pos.from_numpy(mesh.vertices)
    ti_pos_init.from_numpy(mesh.vertices)
    ti_boundary_labels.fill(0)
    set_exforce(exf_angle, exf_mag)
    init_mesh_B(dirichlet, len(dirichlet))
    precomputation()
    lhs_matrix_np = ti_lhs_matrix.to_numpy()
    s_lhs_matrix_np = sparse.csr_matrix(lhs_matrix_np)
    pre_fact_lhs_solve = factorized(s_lhs_matrix_np)

    print("sparse lhs matrix:\n", s_lhs_matrix_np)

    gui = ti.GUI('PN Standalone', background_color=0xf7f7f7)
    wait = input("PRESS ENTER TO CONTINUE.")

    filename = f'./results/frame_rest.png'
    draw_image(gui, filename, ti_pos.to_numpy(), mesh_offset, mesh_scale, ti_f2v.to_numpy(), n_elements)

    frame_counter = 0
    sim_t = 0.0
    plot_array = []

    while frame_counter < 150:
        build_sn()
        # Warm up:
        warm_up()
        print("Frame ", frame_counter)
        last_record_energy = 1000000.0
        for itr in range(solver_max_iteration):
            local_solve_build_bp_for_all_constraints()
            build_rhs(rhs_np)

            local_step_energy = compute_local_step_energy()
            # print("energy after local step:", local_step_energy)
            if local_step_energy > last_record_energy:
                print("Energy Error: LOCAL; Error Amount:",
                      (local_step_energy - last_record_energy) / local_step_energy)
                if (local_step_energy - last_record_energy) / local_step_energy > 0.01:
                    print("Large Error: LOCAL")
            last_record_energy = local_step_energy

            pos_new_np = pre_fact_lhs_solve(rhs_np)
            update_pos_new_from_numpy(pos_new_np)

            global_step_energy = compute_global_step_energy()
            # print("energy after global step:", global_step_energy)
            plot_array.append([itr, global_step_energy])
            if global_step_energy > last_record_energy:
                print("Energy Error: GLOBAL; Error Amount:",
                      (global_step_energy - last_record_energy) / global_step_energy)
                if (global_step_energy - last_record_energy) / global_step_energy > 0.01:
                    print("Large Error: GLOBAL")
            last_record_energy = global_step_energy

        # Update velocity and positions
        update_velocity_pos()
        frame_counter += 1
        filename = f'./results/frame_{frame_counter:05d}.png'
        draw_image(gui, filename, ti_pos.to_numpy(), mesh_offset, mesh_scale, ti_f2v.to_numpy(), n_elements)
        # gui.show(filename)





