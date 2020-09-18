# Acknowledgement: ti example fem99.py

import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

N = 32
dt = 3e-3
# dt = 1.0/480
dx = 1 / N
rho = 4e1
NF = 2 * N ** 2   # number of faces
NV = (N + 1) ** 2  # number of vertices
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.32
gravity = ti.Vector([0, -40])
damping = 12.5
# Area: 0.000061
m_weight_strain = mu * 0.000061 * 2


pos = ti.Vector.field(2, float, NV)
pos_new = ti.Vector.field(2, float, NV)
last_pos_new = ti.Vector.field(2, float, NV)

vel = ti.Vector.field(2, float, NV)
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)  # The inverse of the init elements -- Dm
F = ti.Matrix.field(2, 2, float, NF, needs_grad=True)
A = ti.Matrix.field(4, 6, float, NF)
Bp = ti.Matrix.field(2, 2, float, NF)

rhs = ti.field(float, NV * 2)
rhs_last = ti.field(float, NV * 2)

Sn = ti.field(float, NV * 2)
lhs_matrix = ti.field(ti.f32, shape=(NV * 2, NV * 2))

resolutionX = 512
pixels = ti.var(ti.f32, shape=(resolutionX, resolutionX))

# V = ti.field(float, NF)
# phi = ti.field(float, NF)  # potential energy of each face (Neo-Hookean)
# U = ti.field(float, (), needs_grad=True)  # total potential energy

solver_iteration = 8


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) / N * 0.25 + ti.Vector([0.45, 0.45])
        vel[k] = ti.Vector([0, 0])
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        # print("area:", B_i_inv.determinant())
        B[i] = B_i_inv.inverse()


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]


@ti.kernel
def precomputation():
    # Construct A_i matrix for every element / Build A for all the constraints:
    for i in range(NF):
        # Get (Dm)^-1 for this element:
        Dm_inv_i = B[i]
        a = Dm_inv_i[0, 0]
        b = Dm_inv_i[0, 1]
        c = Dm_inv_i[1, 0]
        d = Dm_inv_i[1, 1]
        # Construct A_i:
        A[i][0, 0] = a
        A[i][0, 2] = c
        A[i][0, 4] = -(a + c)
        A[i][1, 0] = b
        A[i][1, 2] = d
        A[i][1, 4] = -(b + d)
        A[i][2, 1] = a
        A[i][2, 3] = c
        A[i][2, 5] = -(a + c)
        A[i][3, 1] = b
        A[i][3, 3] = d
        A[i][3, 5] = -(b + d)
    # Construct lhs matrix:
    # Init diagonal elements:
    for i in range(NV * 2):
        lhs_matrix[i, i] = rho * 0.000061 / (dt * dt)
    # Map A_i_T * A_i back to this lhs_matrix:
    for ele_idx in range(NF):
        A_i = A[ele_idx]
        ia, ib, ic = f2v[ele_idx]
        ia_x_idx, ia_y_idx = ia * 2, ia * 2 + 1
        ib_x_idx, ib_y_idx = ib * 2, ib * 2 + 1
        ic_x_idx, ic_y_idx = ic * 2, ic * 2 + 1
        q_idx_vec = ti.Vector([ia_x_idx, ia_y_idx, ib_x_idx, ib_y_idx, ic_x_idx, ic_y_idx])
        AT_A = A_i.transpose() @ A_i
        for A_row_idx in ti.static(range(6)):
            for A_col_idx in ti.static(range(6)):
                lhs_matrix_row_idx = q_idx_vec[A_row_idx]
                lhs_matrix_col_idx = q_idx_vec[A_col_idx]
                lhs_matrix[lhs_matrix_row_idx, lhs_matrix_col_idx] += (AT_A[A_row_idx, A_col_idx] * m_weight_strain)


@ti.kernel
def local_solve_build_bp_for_all_constraints():
    for i in range(NF):
        # Construct Current F_i:
        ia, ib, ic = f2v[i]
        a, b, c = pos_new[ia], pos_new[ib], pos_new[ic]
        # V[i] = abs((a - c).cross(b - c))
        # print("[a - c, b - c]:", [a - c, b - c])
        # D_i = ti.Matrix.cols([[1, 0], [0, 1]])
        D_i = ti.Matrix.cols([a - c, b - c])
        F_i = D_i @ B[i]
        # Use current F_i construct current 'B * p' or Ri
        U, sigma, V = ti.svd(F_i, ti.f32)
        Bp[i] = U @ V.transpose()


@ti.kernel
def build_sn():
    for vert_idx in range(NV):
        Sn_idx1 = vert_idx * 2
        Sn_idx2 = vert_idx * 2 + 1
        pos_i = pos[vert_idx]
        vel_i = vel[vert_idx]
        Sn[Sn_idx1] = pos_i[0] + dt * vel_i[0]  # x-direction;
        Sn[Sn_idx2] = pos_i[1] + dt * vel_i[1] + dt * dt * gravity[1]  # y-direction;
    print("Proposed pos:", Sn[0], ", ", Sn[1])


@ti.kernel
def build_rhs():
    one_over_dt2 = 1.0 / (dt ** 2)
    for i in range(NV * 2):
        rhs[i] = one_over_dt2 * rho * 0.000061 * Sn[i]
    for ele_idx in range(NF):
        ia, ib, ic = f2v[ele_idx]
        Bp_i = Bp[ele_idx]  # It is a 2x2 matrix now. We want it be a 4x1 vector.
        Bp_i_vec = ti.Vector([Bp_i[0, 0], Bp_i[0, 1], Bp_i[1, 0], Bp_i[1, 1]])
        A_i = A[ele_idx]
        AT_Bp = A_i.transpose() @ Bp_i_vec  # AT_Bp is a 6x1 vector now.
        AT_Bp *= m_weight_strain
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


@ti.kernel
def update_velocity_pos():
    print("pos_new[0] after solver, before advection:", pos_new[0])
    for i in range(NV):
        vel[i] = (pos_new[i] - pos[i]) / dt
        pos[i] = pos_new[i]
        # rect boundary condition:
        cond = pos[i] < 0.0 and vel[i] < 0 or pos[i] > 1 and vel[i] > 0
        for j in ti.static(range(pos.n)):
            if cond[j]: vel[i][j] = 0.0
        pos[i] += dt * vel[i]
    print("pos[0] after advection:", pos[0])
    print("vel[0] after advection:", vel[0])


@ti.kernel
def warm_up():
    for pos_idx in range(NV):
        sn_idx1, sn_idx2 = pos_idx * 2, pos_idx * 2 + 1
        pos_new[pos_idx][0] = Sn[sn_idx1]
        pos_new[pos_idx][1] = Sn[sn_idx2]


@ti.kernel
def update_pos_new_from_numpy(sol: ti.ext_arr()):
    for pos_idx in range(NV):
        sol_idx1, sol_idx2 = pos_idx * 2, pos_idx * 2 + 1
        pos_new[pos_idx][0] = sol[sol_idx1]
        pos_new[pos_idx][1] = sol[sol_idx2]


@ti.kernel
def check_residual():
    residual = 0.0
    for i in range(NV):
        residual += (last_pos_new[i] - pos_new[i]).norm()
    print("residual:", residual)

frame_counter = 0
init_mesh()
init_pos()
precomputation()
gui = ti.GUI('Projective Dynamics Demo1 v0.1', res=(512, 512), background_color=0xdddddd)
wait = input("PRESS ENTER TO CONTINUE.")
while gui.running:
    build_sn()
    # Warm up:
    warm_up()
    last_pos_new = pos_new
    for itr in range(solver_iteration):
        local_solve_build_bp_for_all_constraints()
        build_rhs()
        # Solve for pos:
        rhs_np = rhs.to_numpy()
        lhs_matrix_np = lhs_matrix.to_numpy()
        pos_new_np = np.linalg.solve(lhs_matrix_np, rhs_np)
        update_pos_new_from_numpy(pos_new_np)
        check_residual()
        last_pos_new = pos_new
    # Update velocity and positions
    update_velocity_pos()

    gui.circles(pos.to_numpy(), radius=2, color=0xffaa33)
    frame_counter += 1
    filename = f'./results/frame_{frame_counter:05d}.png'
    gui.show(filename)
