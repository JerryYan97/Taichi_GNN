# Acknowledgement: ti example fem99.py

import taichi as ti
import numpy as np
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized

ti.init(arch=ti.gpu)

dim = 2
N = 20 # internal of one edge
W = 10
# dt = 3e-3
dt = 1.0/480
dx = 1 / N  # 0.05
rho = 1e1
NF = 2 * N * W # 2 * N ** 2   # number of faces
NV = (N+1)*(W+1) # (N + 1) ** 2 # number of vertices
E, nu = 1e2, 0.4  # Young's modulus and Poisson's ratio
mu, lam = E / (2*(1+nu)), E * nu / ((1+nu)*(1-2*nu))  # Lame parameters
ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.32
gravity = ti.Vector([0, -9.8])
damping = 12.5
# Area: 0.000061 0.02*0.02*sin90*0.5
volume = 0.0002
m_weight_strain = mu * 2 * volume
m_weight_volume = lam * dim * volume
print("m_weight_strain/volume", m_weight_strain/volume, "  m_weight_volume/volume", m_weight_volume/volume)

mass = ti.field(float, NV)

pos = ti.Vector.field(2, float, NV)
pos_new = ti.Vector.field(2, float, NV)
last_pos_new = ti.Vector.field(2, float, NV)
posn = ti.Vector.field(2, float, NV)
boundary_labels = ti.field(int, NV)

vel = ti.Vector.field(2, float, NV)
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)  # The inverse of the init elements -- Dm
F = ti.Matrix.field(2, 2, float, NF, needs_grad=True)
A = ti.Matrix.field(4, 6, float, NF)
Bp = ti.Matrix.field(2, 2, float, NF * 2)
# rhs = ti.field(float, NV * 2)
rhs_np = np.zeros(NV * 2, dtype=np.float32)

Sn = ti.field(float, NV * 2)
# weight = ti.field(float, NF * 2)
lhs_matrix = ti.field(ti.f32, shape=(NV * 2, NV * 2))
phi = ti.field(float, NF)  # potential energy of each element(face) for linear coratated elasticity material.


resolutionX = 512
pixels = ti.var(ti.f32, shape=(resolutionX, resolutionX))

drag = 0.2

# V = ti.field(float, NF)
# phi = ti.field(float, NF)  # potential energy of each face (Neo-Hookean)
# U = ti.field(float, (), needs_grad=True)  # total potential energy

solver_max_iteration = 8
solver_stop_residual = 0.0001


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, W + 1):
        k = i*(W+1)+j
        pos[k] = ti.Vector([i/N*0.4, j/W*0.2]) + ti.Vector([0.2, 0.4]) # 0.2, 0.4 - 0.6,0.6  0.02*0.02
        vel[k] = ti.Vector([0, 0])
        if i == 0:
            boundary_labels[k] = 1
        else:
            boundary_labels[k] = 0
    for i in range(NF): # NF number of face
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([b - a, c - a])  # rest B
        B[i] = B_i_inv.inverse()  # rest of B inverse


@ti.kernel
def init_mesh():  # generate two triangles
    for i, j in ti.ndrange(N, W):
        k = (i * W + j) * 2  # tirangle index   w 2 n 3
        a = i * (W + 1) + j  # 0 0 = 0
        b = a + 1  # 1
        c = a + W + 2  # 12
        d = a + W + 1  # 11
        f2v[k + 0] = [a, d, c]
        f2v[k + 1] = [a, c, b]


@ti.kernel
def precomputation():
    dimp = dim+1
    for e_it in range(NF):
        ia, ib, ic = f2v[e_it]
        mass[ia] += volume/dimp * rho
        mass[ib] += volume/dimp * rho
        mass[ic] += volume/dimp * rho

    # Construct A_i matrix for every element / Build A for all the constraints:
    for t in ti.static(range(2)):
        for i in range(NF):
            # Get (Dm)^-1 for this element:
            Dm_inv_i = B[i]
            a = Dm_inv_i[0, 0]
            b = Dm_inv_i[0, 1]
            c = Dm_inv_i[1, 0]
            d = Dm_inv_i[1, 1]
            # Construct A_i:
            A[t*NF+i][0, 0] = -a-c
            A[t*NF+i][0, 2] = a
            A[t*NF+i][0, 4] = c
            A[t*NF+i][1, 0] = -b-d
            A[t*NF+i][1, 2] = b
            A[t*NF+i][1, 4] = d
            A[t*NF+i][2, 1] = -a-c
            A[t*NF+i][2, 3] = a
            A[t*NF+i][2, 5] = c
            A[t*NF+i][3, 1] = -b-d
            A[t*NF+i][3, 3] = b
            A[t*NF+i][3, 5] = d
    # Construct lhs matrix:
    # Init diagonal elements:
    #for i in range(NV * 2):
    #    lhs_matrix[i,i] = rho * volume / (dt * dt)  # 0.000061
    # Map A_i_T * A_i back to this lhs_matrix:

    for i in range(NV):
        if boundary_labels[i] == 1:
            # Enforce boundary condition:
            sys_idx_x, sys_idx_y = i * 2, i * 2 + 1
            lhs_matrix[sys_idx_x, sys_idx_x] = 1.0
            lhs_matrix[sys_idx_y, sys_idx_y] = 1.0
        else:
            for d in ti.static(range(2)):
                lhs_matrix[i * dim + d, i * dim + d] += (drag / dt) + mass[i] / (dt * dt)


    for t in ti.static(range(2)):
        for ele_idx in range(NF):
            A_i = A[t*NF+ele_idx]
            ia, ib, ic = f2v[ele_idx]
            ia_x_idx, ia_y_idx = ia*2, ia*2+1
            ib_x_idx, ib_y_idx = ib*2, ib*2+1
            ic_x_idx, ic_y_idx = ic*2, ic*2+1
            q_idx_vec = ti.Vector([ia_x_idx, ia_y_idx, ib_x_idx, ib_y_idx, ic_x_idx, ic_y_idx])
            # AT_A = A_i.transpose() @ A_i
            for A_row_idx in ti.static(range(6)):
                for A_col_idx in ti.static(range(6)):
                    lhs_row_idx = q_idx_vec[A_row_idx]
                    lhs_col_idx = q_idx_vec[A_col_idx]
                    vert_idx = ti.floor(lhs_row_idx / 2)
                    # Enforce boundary condition:
                    if boundary_labels[vert_idx] == 0:
                        for idx in ti.static(range(4)):
                            weight = 0.0
                            if t == 0:
                                weight = m_weight_strain
                            else:
                                weight = m_weight_volume
                            lhs_matrix[lhs_row_idx, lhs_col_idx] += (A_i[idx, A_row_idx] * A_i[idx, A_col_idx] * weight)


@ti.kernel
def local_solve_build_bp_for_all_constraints():
    for i in range(NF):
        # Construct strain constraints:
        # Construct Current F_i:
        ia, ib, ic = f2v[i]
        a, b, c = pos_new[ia], pos_new[ib], pos_new[ic]
        D_i = ti.Matrix.cols([b - a, c - a])
        F_i = D_i @ B[i]
        F[i] = F_i
        # Use current F_i construct current 'B * p' or Ri
        U, sigma, V = ti.svd(F_i, ti.f32)
        Bp[i] = U @ V.transpose()

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
        Bp[NF + i] = U @ PP @ V.transpose()

    # Calculate Phi for all the elements:
    for i in range(NF):
        Bp_i_strain = Bp[i]
        Bp_i_volume = Bp[NF + i]
        F_i = F[i]
        energy1 = mu * volume * ((F_i - Bp_i_strain).norm() ** 2)
        energy2 = 0.5 * lam * volume * ((F_i - Bp_i_volume).trace() ** 2)
        phi[i] = energy1 + energy2


@ti.kernel
def build_sn():
    for vert_idx in range(NV):  # number of vertices
        Sn_idx1 = vert_idx*2  # m_sn
        Sn_idx2 = vert_idx*2+1
        pos_i = pos[vert_idx]  # pos = m_x
        # posn[vert_idx] = pos[vert_idx]  # posn = m_xn
        vel_i = vel[vert_idx]
        Sn[Sn_idx1] = pos_i[0] + dt * vel_i[0]  # x-direction;
        Sn[Sn_idx2] = pos_i[1] + dt * vel_i[1] + dt * dt * gravity[1]  # y-direction;
    # print("Proposed pos:", Sn[0], ", ", Sn[1])


@ti.kernel
def build_rhs(rhs: ti.ext_arr()):
    one_over_dt2 = 1.0 / (dt ** 2)
    for i in range(NV * 2):
        # print("test pos ", i/2, " ", i%2)
        # pos_i = posn[i/2]
        pos_i = pos[i/2]
        p0 = pos_i[0]
        p1 = pos_i[1]
        if i % 2 == 0:
            rhs[i] = one_over_dt2 * mass[i/2] * Sn[i] + (drag/dt*p0)  # 0.000061
        else:
            rhs[i] = one_over_dt2 * mass[i/2] * Sn[i] + (drag/dt*p1)  # 0.000061

    for t in ti.static(range(2)):
        for ele_idx in range(NF):
            ia, ib, ic = f2v[ele_idx]
            Bp_i = Bp[t*NF+ele_idx]  # It is a 2x2 matrix now. We want it be a 4x1 vector.
            Bp_i_vec = ti.Vector([Bp_i[0, 0], Bp_i[0, 1], Bp_i[1, 0], Bp_i[1, 1]])
            A_i = A[ele_idx]
            AT_Bp = A_i.transpose() @ Bp_i_vec  # AT_Bp is a 6x1 vector now.
            weight = 0.0
            if t == 0:
                weight = m_weight_strain
            else:
                weight = m_weight_volume
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

    # Enforce boundary condition:
    for i in range(NV):
        pos_i = pos[i]
        if boundary_labels[i] == 1:
            sys_idx_x, sys_idx_y = 2 * i, 2 * i + 1
            rhs[sys_idx_x] = pos_i[0]
            rhs[sys_idx_y] = pos_i[1]


@ti.kernel
def update_velocity_pos():
    for i in range(NV):
        if boundary_labels[i] == 0:
            vel[i] = (pos_new[i] - pos[i]) / dt
            pos[i] = pos_new[i]
        # rect boundary condition:
        # cond = pos[i] < 0.1 and vel[i] < 0 or pos[i] > 1 and vel[i] > 0
        # for j in ti.static(range(pos.n)):
        #     if cond[j]: vel[i][j] = 0.0


@ti.kernel
def warm_up():
    for pos_idx in range(NV):
        sn_idx1, sn_idx2 = pos_idx * 2, pos_idx * 2 + 1
        pos_new[pos_idx][0] = Sn[sn_idx1]
        pos_new[pos_idx][1] = Sn[sn_idx2]


@ti.kernel
def initinfo():
    for i in range(NV):
        if (pos[i][0] > 0.401):
            vel[i][0] = 5
        elif (pos[i][0] < 0.399):
            vel[i][0] = 0
        else:
            vel[i][0] = 0


@ti.kernel
def update_pos_new_from_numpy(sol: ti.ext_arr()):
    for pos_idx in range(NV):
        sol_idx1, sol_idx2 = pos_idx*2, pos_idx*2+1
        pos_new[pos_idx][0] = sol[sol_idx1]
        pos_new[pos_idx][1] = sol[sol_idx2]


@ti.kernel
def check_residual() -> ti.f32:
    residual = 0.0
    for i in range(NV):
        residual += (last_pos_new[i] - pos_new[i]).norm()
        last_pos_new[i] = pos_new[i]
    # print("residual:", residual)
    return residual


@ti.kernel
def check_boundary_points():
    # Enforce boundary condition:
    counter = 0
    for i in range(NV):
        pos_i = pos[i]
        if abs(pos_i[0] - 0.2) < 0.001:
            counter += 1
    print("boundary points number:", counter)


def paint_phi(gui):
    pos_np = pos.to_numpy()
    phi_np = phi.to_numpy()
    f2v_np = f2v.to_numpy()
    a, b, c = pos_np[f2v_np[:, 0]], pos_np[f2v_np[:, 1]], pos_np[f2v_np[:, 2]]
    k = phi_np * (8000 / E)
    gb = (1 - k) * 0.7
    # gb = 0.5
    # print("gb:", gb[0])
    # print("phi_np", phi_np[0])
    # print("k", k[0])
    gui.triangles(a, b, c, color=ti.rgb_to_hex([k + gb, gb, gb]))
    gui.lines(a, b, color=0xffffff, radius=0.5)
    gui.lines(b, c, color=0xffffff, radius=0.5)
    gui.lines(c, a, color=0xffffff, radius=0.5)


frame_counter = 0
init_mesh()
init_pos()
precomputation()
lhs_matrix_np = lhs_matrix.to_numpy()
s_lhs_matrix_np = sparse.csr_matrix(lhs_matrix_np)
pre_fact_lhs_solve = factorized(s_lhs_matrix_np)

print("sparse lhs matrix:\n", s_lhs_matrix_np)

initinfo()

gui = ti.GUI('Projective Dynamics Demo3 v0.2')
wait = input("PRESS ENTER TO CONTINUE.")

gui.circles(pos.to_numpy(), radius=2, color=0xffaa33)
filename = f'./results/frame_rest.png'
gui.show(filename)

while gui.running:
    build_sn()
    # Warm up:
    warm_up()
    for itr in range(solver_max_iteration):

        # start_solve_constraints_time = time.perf_counter_ns()
        local_solve_build_bp_for_all_constraints()
        # end_solve_constraints_time = time.perf_counter_ns()
        # print("solve constraints time elapsed:", end_solve_constraints_time - start_solve_constraints_time)

        # start_build_rhs_time = time.perf_counter_ns()
        build_rhs(rhs_np)
        # end_build_rhs_time = time.perf_counter_ns()
        # print("build rhs time elapsed:", end_build_rhs_time - start_build_rhs_time)

        # start_linear_solve_time = time.perf_counter_ns()
        pos_new_np = pre_fact_lhs_solve(rhs_np)
        # end_linear_solve_time = time.perf_counter_ns()
        # print("linear solve time elapsed:", end_linear_solve_time - start_linear_solve_time)

        # start_update_pos_time = time.perf_counter_ns()
        update_pos_new_from_numpy(pos_new_np)
        # end_update_pos_time = time.perf_counter_ns()
        # print("update pos new elapsed:", end_update_pos_time - start_update_pos_time)

        # start_check_residual_time = time.perf_counter_ns()
        residual = check_residual()
        # end_check_residual_time = time.perf_counter_ns()
        # print("check residual elapsed:", end_check_residual_time - start_check_residual_time)

        # check_boundary_points()
        if residual < solver_stop_residual:
            break

    # Update velocity and positions
    update_velocity_pos()
    paint_phi(gui)
    gui.circles(pos.to_numpy(), radius=2, color=0xd1d1d1)
    frame_counter += 1
    filename = f'./results/frame_{frame_counter:05d}.png'
    gui.show(filename)
    # print("\n")


# Performance note (unit: ns):
# # solve constraints time elapsed: 54000
# # build rhs time elapsed: 324600
# # linear solve time elapsed: 35200
# # check residual elapsed: 502900
# update pos new elapsed: 189100

# solve constraints time elapsed: 57900
# build rhs time elapsed: 291500
# linear solve time elapsed: 2013200
# check residual elapsed: 501900
# update pos new elapsed: 173600

# solve time elapsed: 16916100
# check residual elapsed: 689100
# update pos new elapsed: 334500
# solve constraints time elapsed: 63200
# build rhs time elapsed: 398600

