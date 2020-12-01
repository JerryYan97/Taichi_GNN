import taichi as ti
import numpy as np
import sys, os, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import csv
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from Utils.reader import *
import pymesh
import math
from .PN import *
from numpy.linalg import inv
from scipy.linalg import sqrtm
from numpy import linalg as LA
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)
real = ti.f64


def calcR(A_pq):
    S = sqrtm(np.dot(np.transpose(A_pq), A_pq))
    # print("S: \n", S)
    R = np.dot(A_pq, inv(S))
    return R


@ti.data_oriented
class PDSimulation:
    def __init__(self, obj_file, _dim):
        self.dim = _dim
        self.dt = 0.01

        ################################ mesh ######################################
        self.mesh, _, _, _ = read(int(obj_file))
        self.NV, self.NF, _ = self.mesh.num_vertices, self.mesh.num_faces, self.mesh.num_voxels
        self.n_particles = self.NV
        ################################ material ######################################
        self.rho = 100
        self.E = 1e4
        self.nu = 0.4
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        ################################ field ######################################
        self.volume = ti.field(real, self.NF)
        self.m_weight_strain = ti.field(real, self.NF)  # self.mu * 2 * self.volume
        self.m_weight_volume = ti.field(real, self.NF)  # self.lam * self.dim * self.volume

        self.m_weight_positional = 10000000.0

        self.mass = ti.field(ti.f64, self.NV)
        self.pos = ti.Vector.field(2, ti.f64, self.NV)
        self.pos_new = ti.Vector.field(2, ti.f64, self.NV)
        self.pos_init = ti.Vector.field(2, ti.f64, self.NV)
        self.last_pos_new = ti.Vector.field(2, ti.f64, self.NV)
        self.boundary_labels = ti.field(int, self.NV)

        self.pos_delta = ti.var(real, shape=2 * self.NV)
        self.pos_delta2 = ti.Vector.field(2, real, self.NV)

        self.vel = ti.Vector.field(2, ti.f64, self.NV)
        self.vel_last = ti.Vector.field(2, ti.f64, self.NV)

        self.f2v = ti.Vector.field(3, int, self.NF)  # ids of three vertices of each face
        self.B = ti.Matrix.field(2, 2, ti.f64, self.NF)  # The inverse of the init elements -- Dm
        self.F = ti.Matrix.field(2, 2, ti.f64, self.NF, needs_grad=True)
        self.A = ti.Matrix.field(4, 6, ti.f64, self.NF * 2)
        self.Bp = ti.Matrix.field(2, 2, ti.f64, self.NF * 2)

        self.rhs_np = np.zeros(self.NV * 2, dtype=np.float64)
        self.Sn = ti.field(ti.f64, self.NV * 2)

        self.lhs_matrix = ti.field(ti.f64, shape=(self.NV * 2, self.NV * 2))
        self.phi = ti.field(ti.f64, self.NF)

        self.resolutionX = 512
        self.pixels = ti.var(ti.f32, shape=(self.resolutionX, self.resolutionX))

        self.drag = 0.0  # 0.2

        self.solver_max_iteration = 10
        self.solver_stop_residual = 0.0001

        self.gradE = ti.field(real, shape=2000)
        self.x_xtilde = ti.Vector.field(self.dim, real, self.n_particles)

        ################################ shape matching ######################################
        self.initial_com = ti.Vector([0.0, 0.0])
        self.initial_rel_pos = np.array([self.n_particles, 2])

        self.pi = ti.Vector.field(self.dim, real, self.n_particles)
        self.qi = ti.Vector.field(self.dim, real, self.n_particles)
        self.init_pos = self.mesh.vertices.astype(np.float32)
        self.init_pos = self.init_pos[:, :2]

        ################################ force field ################################
        self.gravity = ti.Vector([0, 0])
        self.exf_angle = np.arange(0, 2 * np.pi, 30)
        self.exf_mag = np.arange(0, 10.0, 30)
        # print("angle: ", self.exf_angle, " mag: ", self.exf_mag)
        self.exf_ind = 0
        self.mag_ind = 0
        self.ex_force = ti.Vector.field(self.dim, real, 1)
        self.npex_f = np.zeros((2, 1))

        ################################ boundary setting ################################
        edges = set()
        for [i, j, k] in self.mesh.faces:
            edges.add((i, j))
            edges.add((j, k))
            edges.add((k, i))
        self.boundary_points_ = set()
        for [i, j, k] in self.mesh.faces:
            if (j, i) not in edges:
                self.boundary_points_.update([j, i])
            if (k, j) not in edges:
                self.boundary_points_.update([k, j])
            if (i, k) not in edges:
                self.boundary_points_.update([i, k])

        self.input_xn = ti.Vector.field(self.dim, real, self.NV)
        self.input_vn = ti.Vector.field(self.dim, real, self.NV)

    def set_material(self, _rho, _ym, _nu, _dt):
        self.dt = _dt
        self.rho = _rho
        self.E = _ym
        self.nu = _nu
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    def set_force(self, ind, mag):
        self.exf_ind = ind
        self.mag_ind = mag
        x = 0.3 * mag * ti.sin(3.1415926 / 30.0 * ind)
        y = 0.3 * mag * ti.cos(3.1415926 / 30.0 * ind)
        self.ex_force[0] = ti.Vector([x, y])

    def init_mesh_obj(self):
        for i in range(self.mesh.num_faces):
            self.f2v[i] = ti.Vector([self.mesh.faces[i][0], self.mesh.faces[i][1], self.mesh.faces[i][2]])
        for i in range(self.mesh.num_vertices):
            self.pos[i] = ti.Vector([self.mesh.vertices[i][0], self.mesh.vertices[i][1]])
            self.pos_init[i] = ti.Vector([self.mesh.vertices[i][0], self.mesh.vertices[i][1]])
            self.vel[i] = ti.Vector([0, 0])
            self.vel_last[i] = ti.Vector([0, 0])
            if i in self.boundary_points_ and i <= 10:
                self.boundary_labels[i] = 1
            else:
                self.boundary_labels[i] = 0

    @ti.kernel
    def init_mesh_B(self):
        for i in range(self.NF):  # NF number of face
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos_init[ia], self.pos_init[ib], self.pos_init[ic]
            B_i_inv = ti.Matrix.cols([b - a, c - a])  # rest B
            self.B[i] = B_i_inv.inverse()  # rest of B inverse
            self.volume[i] = B_i_inv.determinant() * 0.5
            self.m_weight_strain[i] = self.mu * 2 * self.volume[i]
            self.m_weight_volume[i] = self.lam * self.dim * self.volume[i]

    @ti.kernel
    def precomputation(self):
        dimp = self.dim + 1
        for e_it in range(self.NF):
            ia, ib, ic = self.f2v[e_it]
            self.mass[ia] += self.volume[e_it] / dimp * self.rho
            self.mass[ib] += self.volume[e_it] / dimp * self.rho
            self.mass[ic] += self.volume[e_it] / dimp * self.rho
        # Construct A_i matrix for every element / Build A for all the constraints:
        # Strain constraints and area constraints
        for t in ti.static(range(2)):
            for i in range(self.NF):
                Dm_inv_i = self.B[i]  # Get (Dm)^-1 for this element:
                a = Dm_inv_i[0, 0]
                b = Dm_inv_i[0, 1]
                c = Dm_inv_i[1, 0]
                d = Dm_inv_i[1, 1]
                # Construct A_i:
                self.A[t * self.NF + i][0, 0] = -a - c
                self.A[t * self.NF + i][0, 2] = a
                self.A[t * self.NF + i][0, 4] = c
                self.A[t * self.NF + i][1, 0] = -b - d
                self.A[t * self.NF + i][1, 2] = b
                self.A[t * self.NF + i][1, 4] = d
                self.A[t * self.NF + i][2, 1] = -a - c
                self.A[t * self.NF + i][2, 3] = a
                self.A[t * self.NF + i][2, 5] = c
                self.A[t * self.NF + i][3, 1] = -b - d
                self.A[t * self.NF + i][3, 3] = b
                self.A[t * self.NF + i][3, 5] = d

        # Construct lhs matrix without constraints
        for i in range(self.NV):
            for d in ti.static(range(2)):
                self.lhs_matrix[i * self.dim + d, i * self.dim + d] = (self.drag / self.dt) + self.mass[i] / (
                            self.dt * self.dt)

        # Add strain and area/volume constraints to the lhs matrix
        for t in ti.static(range(2)):
            for ele_idx in range(self.NF):
                A_i = self.A[t * self.NF + ele_idx]
                ia, ib, ic = self.f2v[ele_idx]
                ia_x_idx, ia_y_idx = ia * 2, ia * 2 + 1
                ib_x_idx, ib_y_idx = ib * 2, ib * 2 + 1
                ic_x_idx, ic_y_idx = ic * 2, ic * 2 + 1
                q_idx_vec = ti.Vector([ia_x_idx, ia_y_idx, ib_x_idx, ib_y_idx, ic_x_idx, ic_y_idx])
                # AT_A = A_i.transpose() @ A_i
                for A_row_idx in ti.static(range(6)):
                    for A_col_idx in ti.static(range(6)):
                        lhs_row_idx = q_idx_vec[A_row_idx]
                        lhs_col_idx = q_idx_vec[A_col_idx]
                        for idx in ti.static(range(4)):
                            weight = 0.0
                            if t == 0:
                                weight = self.m_weight_strain[ele_idx]
                            else:
                                weight = self.m_weight_volume[ele_idx]
                            self.lhs_matrix[lhs_row_idx, lhs_col_idx] += (
                                        A_i[idx, A_row_idx] * A_i[idx, A_col_idx] * weight)

        # Add positional constraints to the lhs matrix
        for i in range(self.NV):
            if self.boundary_labels[i] == 1:
                q_i_x_idx = i * 2
                q_i_y_idx = i * 2 + 1
                self.lhs_matrix[
                    q_i_x_idx, q_i_x_idx] += self.m_weight_positional  # This is the weight of positional constraints
                self.lhs_matrix[q_i_y_idx, q_i_y_idx] += self.m_weight_positional

    # NOTE: This function doesn't build all constraints
    # It just builds strain constraints and area/volume constraints
    @ti.kernel
    def local_solve_build_bp_for_all_constraints(self):
        for i in range(self.NF):
            # Construct strain constraints:
            # Construct Current F_i:
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos_new[ia], self.pos_new[ib], self.pos_new[ic]
            D_i = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(D_i @ self.B[i], ti.f64)
            self.F[i] = F_i
            # Use current F_i construct current 'B * p' or Ri
            U, sigma, V = ti.svd(F_i, ti.f64)
            self.Bp[i] = U @ V.transpose()
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
            self.Bp[self.NF + i] = U @ PP @ V.transpose()

        # Calculate Phi for all the elements:
        for i in range(self.NF):
            Bp_i_strain = self.Bp[i]
            Bp_i_volume = self.Bp[self.NF + i]
            F_i = self.F[i]
            energy1 = self.mu * self.volume[i] * ((F_i - Bp_i_strain).norm() ** 2)
            energy2 = 0.5 * self.lam * self.volume[i] * ((F_i - Bp_i_volume).trace() ** 2)
            self.phi[i] = energy1 + energy2

    @ti.kernel
    def build_sn(self):
        for vert_idx in range(self.NV):  # number of vertices
            Sn_idx1 = vert_idx * 2  # m_sn
            Sn_idx2 = vert_idx * 2 + 1
            pos_i = self.pos[vert_idx]  # pos = m_x
            vel_i = self.vel[vert_idx]
            self.Sn[Sn_idx1] = pos_i[0] + self.dt * vel_i[0] + (self.dt * self.dt) * (
                        self.ex_force[0][0] / self.mass[vert_idx])  # x-direction;
            self.Sn[Sn_idx2] = pos_i[1] + self.dt * vel_i[1] + (self.dt * self.dt) * (
                        self.ex_force[0][1] / self.mass[vert_idx])  # y-direction;

    @ti.kernel
    def compute_x_xtilde(self):
        for i in range(self.n_particles):
            self.x_xtilde[i][0] = self.pos[i][0] - self.Sn[i * 2]
            self.x_xtilde[i][1] = self.pos[i][1] - self.Sn[i * 2 + 1]
            # print("x-xtilde: ", self.x_xtilde[i], " x: ", self.pos[i], " sn: ", self.Sn[i*2], ", ", self.Sn[i*2+1])

    @ti.kernel
    def build_rhs(self, rhs: ti.ext_arr()):
        one_over_dt2 = 1.0 / (self.dt ** 2)
        for i in range(self.NV * 2):  # Construct the first part of the rhs
            pos_i = self.pos[i / 2]
            p0 = pos_i[0]
            p1 = pos_i[1]
            if i % 2 == 0:
                rhs[i] = one_over_dt2 * self.mass[i / 2] * self.Sn[i] + (self.drag / self.dt * p0)  # 0.000061
            else:
                rhs[i] = one_over_dt2 * self.mass[i / 2] * self.Sn[i] + (self.drag / self.dt * p1)  # 0.000061
        # Add strain and volume/area constraints to the rhs
        for t in ti.static(range(2)):
            for ele_idx in range(self.NF):
                ia, ib, ic = self.f2v[ele_idx]
                Bp_i = self.Bp[t * self.NF + ele_idx]  # It is a 2x2 matrix now. We want it be a 4x1 vector.
                Bp_i_vec = ti.Vector([Bp_i[0, 0], Bp_i[0, 1], Bp_i[1, 0], Bp_i[1, 1]])
                A_i = self.A[ele_idx]
                AT_Bp = A_i.transpose() @ Bp_i_vec  # AT_Bp is a 6x1 vector now.
                weight = 0.0
                if t == 0:
                    weight = self.m_weight_strain[ele_idx]
                else:
                    weight = self.m_weight_volume[ele_idx]
                AT_Bp *= weight  # m_weight_strain
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
        # Add positional constraints Bp to the rhs
        for i in range(self.NV):
            if self.boundary_labels[i] == 1:
                pos_init_i = self.pos_init[i]
                q_i_x_idx = i * 2
                q_i_y_idx = i * 2 + 1
                rhs[q_i_x_idx] += (pos_init_i[0] * self.m_weight_positional)
                rhs[q_i_y_idx] += (pos_init_i[1] * self.m_weight_positional)

    @ti.kernel
    def update_velocity_pos(self):
        for i in range(self.NV):
            self.vel_last[i] = self.vel[i]
            self.vel[i] = (self.pos_new[i] - self.pos[i]) / self.dt
            self.pos_delta2[i] = self.pos_new[i] - self.pos[i]
            self.pos[i] = self.pos_new[i]
            # print("vel last: ", self.vel_last[i], "vel: ", self.vel[i])

    @ti.kernel
    def update_delta_pos(self):
        for i in range(self.NV):
            if self.boundary_labels[i] == 0:
                self.pos_delta[i] = self.pos_new[i][0] - self.pos[i][0]
                self.pos_delta[i + 1] = self.pos_new[i][1] - self.pos[i][1]
                # self.pos_delta2[i] = self.pos_new[i] - self.pos[i]

    @ti.kernel
    def warm_up(self):
        for pos_idx in range(self.NV):
            sn_idx1, sn_idx2 = pos_idx * 2, pos_idx * 2 + 1
            self.pos_new[pos_idx][0] = self.Sn[sn_idx1]
            self.pos_new[pos_idx][1] = self.Sn[sn_idx2]

    @ti.kernel
    def initinfo(self):
        for i in range(self.NV):
            if (self.pos[i][0] > 0.401):
                self.vel[i][0] = 5
            elif (self.pos[i][0] < 0.399):
                self.vel[i][0] = 0
            else:
                self.vel[i][0] = 0

    @ti.kernel
    def update_pos_new_from_numpy(self, sol: ti.ext_arr()):
        for pos_idx in range(self.NV):
            sol_idx1, sol_idx2 = pos_idx * 2, pos_idx * 2 + 1
            self.pos_new[pos_idx][0] = sol[sol_idx1]
            self.pos_new[pos_idx][1] = sol[sol_idx2]

    @ti.kernel
    def check_residual(self) -> ti.f32:
        residual = 0.0
        for i in range(self.NV):
            residual += (self.last_pos_new[i] - self.pos_new[i]).norm()
            self.last_pos_new[i] = self.pos_new[i]
        # print("residual:", residual)
        return residual

    @ti.kernel
    def compute_T1_energy(self) -> ti.f64:
        T1 = 0.0
        for i in range(self.NV):
            sn_idx1, sn_idx2 = i * 2, i * 2 + 1
            sn_i = ti.Vector([self.Sn[sn_idx1], self.Sn[sn_idx2]])
            temp_diff = (self.pos_new[i] - sn_i) * ti.sqrt(self.mass[i])
            T1 += (temp_diff[0] ** 2 + temp_diff[1] ** 2)
        return T1 / (2.0 * self.dt ** 2)

    @ti.kernel
    def global_compute_T2_energy(self) -> ti.f64:
        T2_global_energy = ti.cast(0.0, ti.f64)
        # Calculate the energy contributed by strain and volume/area constraints
        for i in range(self.NF):
            # Construct Current F_i
            ia, ib, ic = self.f2v[i]
            a, b, c = self.pos_new[ia], self.pos_new[ib], self.pos_new[ic]
            D_i = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(D_i @ self.B[i], ti.f64)
            # Get current Bp
            Bp_i_strain = self.Bp[i]
            Bp_i_volume = self.Bp[self.NF + i]
            energy1 = self.m_weight_strain[i] * ((F_i - Bp_i_strain).norm() ** 2) / ti.cast(2.0, ti.f64)
            energy2 = self.m_weight_volume[i] * ((F_i - Bp_i_volume).norm() ** 2) / ti.cast(2.0, ti.f64)
            T2_global_energy += (energy1 + energy2)
        # Calculate the energy contributed by positional constraints
        # total_energy3 = 0.0
        for i in range(self.NV):
            if self.boundary_labels[i] == 1:
                pos_init_i = self.pos_init[i]
                pos_curr_i = self.pos_new[i]
                energy3 = self.m_weight_positional * ((pos_curr_i - pos_init_i).norm() ** 2) / ti.cast(2.0, ti.f64)
                # total_energy3 += energy3
                T2_global_energy += energy3
        # print("global energy3:", total_energy3)
        return T2_global_energy

    @ti.kernel
    def local_compute_T2_energy(self) -> ti.f64:
        local_T2_energy = ti.cast(0.0, ti.f64)  # Calculate T2 energy
        # Calculate the energy contributed by strain and volume/area constraints
        for e_it in range(self.NF):
            Bp_i_strain = self.Bp[e_it]
            Bp_i_volume = self.Bp[e_it + self.NF]
            F_i = self.F[e_it]
            energy1 = self.m_weight_strain[e_it] * ((F_i - Bp_i_strain).norm() ** 2) / ti.cast(2.0, ti.f64)
            energy2 = self.m_weight_volume[e_it] * ((F_i - Bp_i_volume).norm() ** 2) / ti.cast(2.0, ti.f64)
            local_T2_energy += (energy1 + energy2)
        # Calculate the energy contributed by positional constraints
        # total_energy3 = 0.0
        for i in range(self.NV):
            if self.boundary_labels[i] == 1:
                pos_init_i = self.pos_init[i]
                pos_curr_i = self.pos_new[i]
                energy3 = self.m_weight_positional * ((pos_curr_i - pos_init_i).norm() ** 2) / ti.cast(2.0, ti.f64)
                # total_energy3 += energy3
                local_T2_energy += energy3
        # print("local energy3:", total_energy3)
        return local_T2_energy

    def compute_global_step_energy(self):
        global_T2_energy = self.global_compute_T2_energy()  # Calculate global T2 energy
        global_T1_energy = self.compute_T1_energy()  # Calculate global T1 energy
        return (global_T1_energy + global_T2_energy)

    def compute_local_step_energy(self):
        local_T2_energy = self.local_compute_T2_energy()
        local_T1_energy = self.compute_T1_energy()  # Calculate T1 energy
        return (local_T1_energy + local_T2_energy)

    def paint_phi(self, gui):
        pos_np = self.pos.to_numpy()
        phi_np = self.phi.to_numpy()
        f2v_np = self.f2v.to_numpy()
        a, b, c = pos_np[f2v_np[:, 0]], pos_np[f2v_np[:, 1]], pos_np[f2v_np[:, 2]]
        k = phi_np * (8000 / self.E)
        gb = (1 - k) * 0.7
        gui.triangles(a, b, c, color=ti.rgb_to_hex([k + gb, gb, gb]))
        gui.lines(a, b, color=0xffffff, radius=0.5)
        gui.lines(b, c, color=0xffffff, radius=0.5)
        gui.lines(c, a, color=0xffffff, radius=0.5)

    @ti.kernel
    def copy(self, x: ti.template(), y: ti.template()):
        for i in x:
            y[i] = x[i]

    ################################### K means part #####################################
    def get_mesh_map(self, mesh):
        map = np.zeros((mesh.num_vertices, mesh.num_vertices))
        mesh.enable_connectivity()
        for p in range(mesh.num_vertices):
            adj_v = mesh.get_vertex_adjacent_vertices(p)
            for j in range(adj_v.shape[0]):
                n1 = p
                n2 = adj_v[j]
                p1 = self.init_pos[n1, :]
                p2 = self.init_pos[n2, :]
                dp = LA.norm(p1 - p2)
                map[n1][n2] = map[n2][n1] = dp
        map_list = map.tolist()
        return map_list

    def update_centers(self, center_pos, parent_list, child_list, belonging):
        delta_list = []
        for p in range(len(center_pos)):
            c_pos = center_pos[p]
            c_id_list = [i for i, x in enumerate(belonging) if x == parent_list[p]]
            count = belonging.count(parent_list[p]) + 1
            sum = c_pos
            for c in c_id_list:
                sum = np.add(sum, self.init_pos[child_list[c], :])
            sum = sum / (1.0 * count)
            min_dis = 100000.0
            new_c = -1
            for c in child_list:
                dis = np.linalg.norm(np.subtract(sum, self.init_pos[c, :]))
                if dis < min_dis:
                    min_dis = dis
                    new_c = c
            dis2 = np.linalg.norm(np.subtract(sum, c_pos))
            if min_dis < dis2:
                center_pos[p] = self.init_pos[new_c, :]
                parent_list[p] = new_c
            delta = np.linalg.norm(np.subtract(center_pos[p], c_pos))
            delta_list.append(delta)
        return center_pos, np.linalg.norm(delta_list), parent_list

    def K_means(self, mesh, k):
        center_pos = []
        whole_list = [n for n in range(0, mesh.num_vertices)]
        parent_list = random.sample(range(0, mesh.num_vertices), k)
        child_list = [x for x in whole_list if x not in parent_list]
        belonging = [None] * len(child_list)  # length: child
        for p in parent_list:
            center_pos.append(self.init_pos[p, :])
        norm_d = 10000.0
        map_list = self.get_mesh_map(mesh)
        Graph = Dijkstra(mesh.num_vertices, map_list, False)
        while norm_d > 1.0:
            t = 0
            for item1 in child_list:
                min_dis = 100000.0
                parent_id = -1
                for p in parent_list:
                    dis = Graph.dijkstra2node(item1, p)
                    if dis < min_dis:
                        parent_id = p
                        min_dis = dis
                belonging[t] = parent_id
                t = t + 1
            center_pos, norm_d, parent_list = self.update_centers(center_pos, parent_list, child_list, belonging)
            child_list = [x for x in whole_list if x not in parent_list]  # update child
        t = 0
        for item1 in child_list:
            min_dis = 100000.0
            parent_id = -1
            for p in parent_list:
                dis = Graph.dijkstra2node(item1, p)
                if dis < min_dis:
                    parent_id = p
                    min_dis = dis
            belonging[t] = parent_id
            t = t + 1

        return center_pos, child_list, parent_list, belonging

    def draw_graph(self, k):
        _, child_list, parent_list, belonging = self.K_means(self.mesh, k)
        color_tab = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:olive', 'tab:gray', 'tab:cyan',
                     'tab:pink', 'tab:red', 'tab:brown']
        for i in range(k):
            posidx = np.where(np.asarray(belonging) == parent_list[i])[0]
            pos = self.init_pos[np.asarray(child_list)[posidx]]
            x = [item[0] for item in pos]
            x.append(self.init_pos[parent_list[i], 0])
            y = [item[1] for item in pos]
            y.append(self.init_pos[parent_list[i], 1])
            plt.scatter(x, y, label="stars", color=color_tab[i], marker="*", s=30)
        plt.xlabel('x - axis')  # x-axis label
        plt.ylabel('y - axis')  # frequency label
        plt.title('K-means result plot!')  # plot title
        plt.legend()  # showing legend
        plt.show()  # function to show the plot
        plt.savefig('kmeans_result.png')

    ################################### shape matching #####################################
    def calcCenterOfMass(self, vind):
        sum = ti.Vector([0.0, 0.0])
        summ = 0.0
        for i in vind:
            for d in ti.static(range(2)):
                sum[d] += self.mass[i] * self.pos[i][d]
            summ += self.mass[i]
        sum[0] /= summ
        sum[1] /= summ
        return sum

    def calcA_qq(self, q_i):
        sum = np.zeros((2, 2))
        for i in range(q_i.shape[0]):
            sum += np.outer(q_i[i], np.transpose(q_i[i]))
        return np.linalg.inv(sum)

    def calcA_pq(self, p_i, q_i):
        sum = np.zeros((2, 2))
        for i in range(p_i.shape[0]):
            sum += np.outer(p_i[i], np.transpose(q_i[i]))
        return sum

    def calcR(self, A_pq):
        S = sqrtm(np.dot(np.transpose(A_pq), A_pq))
        R = np.dot(A_pq, inv(S))
        return R

    def build_pos_arr(self, adj_v, arr, rel_pos):
        result = np.zeros((adj_v.shape[0], self.dim))
        result_pos = np.zeros((adj_v.shape[0], self.dim))
        t = 0
        nparr = arr.to_numpy()
        for p in adj_v:  # print("print! ", nparr[p, 0])
            result[t, 0] = nparr[p, 0]
            result[t, 1] = nparr[p, 1]
            result_pos[t, :] = rel_pos[p, :]
            t = t + 1
        return result, result_pos

    def data_one_frame(self, input_p, input_v):
        lhs_matrix_np = self.lhs_matrix.to_numpy()
        s_lhs_matrix_np = sparse.csr_matrix(lhs_matrix_np)
        pre_fact_lhs_solve = factorized(s_lhs_matrix_np)
        self.copy(input_p, self.pos)
        self.copy(input_v, self.vel)
        self.build_sn()
        self.warm_up()
        last_record_energy = 100000000000.0
        for itr in range(self.solver_max_iteration):
            self.local_solve_build_bp_for_all_constraints()
            self.build_rhs(self.rhs_np)
            local_step_energy = self.compute_local_step_energy()
            if local_step_energy > last_record_energy:
                print("Energy Error: LOCAL; Error Amount:",
                      (local_step_energy - last_record_energy) / local_step_energy)
                if (local_step_energy - last_record_energy) / local_step_energy > 0.01:
                    print("Large Error: LOCAL")
            last_record_energy = local_step_energy
            pos_new_np = pre_fact_lhs_solve(self.rhs_np)
            self.update_pos_new_from_numpy(pos_new_np)
            global_step_energy = self.compute_global_step_energy()
            if global_step_energy > last_record_energy:
                print("Energy Error: GLOBAL; Error Amount:",
                      (global_step_energy - last_record_energy) / global_step_energy)
                if (global_step_energy - last_record_energy) / global_step_energy > 0.01:
                    print("Large Error: GLOBAL")
            last_record_energy = global_step_energy
        # Update velocity and positions
        self.update_velocity_pos()
        return self.pos_delta2, self.pos_new, self.vel

    def compute_restT_and_m(self):
        self.init_mesh_obj()
        self.init_mesh_B()  # this is only for the rest position, so it is ok!
        self.mass.fill(0.0)
        self.lhs_matrix.fill(0.0)
        self.precomputation()

    def get_local_transformation(self):
        ele_count = self.dim * self.dim
        out = np.ones([self.n_particles, ele_count], dtype=float)
        self.mesh.enable_connectivity()
        for i in range(self.n_particles):
            adj_v = self.mesh.get_vertex_adjacent_vertices(i)
            adj_v = np.append(adj_v, i)
            adj_v = np.sort(adj_v)
            new_pos, init_rel_pos = self.build_pos_arr(adj_v, self.pos, self.initial_rel_pos)
            com = self.calcCenterOfMass(adj_v).to_numpy()
            curr_rel_pos = np.zeros((adj_v.shape[0], self.dim))
            for j in range(new_pos.shape[0]):
                curr_rel_pos[j, :] = new_pos[j, :] - com
            A_pq = self.calcA_pq(curr_rel_pos, init_rel_pos)
            R = self.calcR(A_pq)
            out[i, 0] = R[0, 0]
            out[i, 1] = R[0, 1]
            out[i, 2] = R[1, 0]
            out[i, 3] = R[1, 1]
        return out

    def output_all(self, pd_dis, pn_dis, grad_E, frame, T):
        frame = str(frame).zfill(2)
        if T == 0:
            out_name = "Outputs/output" + str(self.exf_ind) + "_" + str(self.mag_ind) + "_" + frame + ".csv"
        else:
            out_name = "Outputs_T/output" + str(self.exf_ind) + "_" + str(self.mag_ind) + "_" + frame + ".csv"
        if not os.path.exists(out_name):
            file = open(out_name, 'w+')
            file.close()
        ele_count = self.dim + self.dim + self.dim * self.dim + self.dim + self.dim + self.dim + self.dim  # pd pos + pn pos + local transform + residual + force
        out = np.ones([self.n_particles, ele_count], dtype=float)
        self.mesh.enable_connectivity()
        for i in range(self.n_particles):
            out[i, 0] = pd_dis[i, 0]  # pd pos
            out[i, 1] = pd_dis[i, 1]
            out[i, 2] = pn_dis[i, 0]  # pn pos
            out[i, 3] = pn_dis[i, 1]
            adj_v = self.mesh.get_vertex_adjacent_vertices(i)
            adj_v = np.append(adj_v, i)
            adj_v = np.sort(adj_v)
            new_pos, init_rel_pos = self.build_pos_arr(adj_v, self.pos, self.initial_rel_pos)
            com = self.calcCenterOfMass(adj_v).to_numpy()
            curr_rel_pos = np.zeros((adj_v.shape[0], self.dim))
            for j in range(new_pos.shape[0]):
                curr_rel_pos[j, :] = new_pos[j, :] - com
            A_pq = self.calcA_pq(curr_rel_pos, init_rel_pos)
            A_qq = self.calcA_qq(init_rel_pos)
            A_final = np.matmul(A_pq, A_qq)
            # TODO： Change to deformation gradient A.
            out[i, 4] = A_final[0, 0]
            out[i, 5] = A_final[0, 1]
            out[i, 6] = A_final[1, 0]
            out[i, 7] = A_final[1, 1]
            out[i, 8] = grad_E[i * 2]
            out[i, 9] = grad_E[i * 2 + 1]
            out[i, 10] = self.ex_force[0][0]
            out[i, 11] = self.ex_force[0][1]
            # temp_f = np.array([self.mass[i] * self.x_xtilde[i][0], self.mass[i] * self.x_xtilde[i][1]])
            # self.npex_f[0] = self.ex_force[0][0]
            # self.npex_f[1] = self.ex_force[0][1]
            # out[i, 10] = self.npex_f[0] + temp_f[0]  # output_deformation_gradient
            # out[i, 11] = self.npex_f[1] + temp_f[1]
            out[i, 12] = self.vel[i][0]  # velocity
            out[i, 13] = self.vel[i][1]
            out[i, 14] = self.pos_init[i][0]  # rest shape
            out[i, 15] = self.pos_init[i][1]
        np.savetxt(out_name, out, delimiter=',')

    def Run(self, pn, is_test, frame_count):
        self.init_mesh_obj()
        self.init_mesh_B()
        self.mass.fill(0.0)
        self.lhs_matrix.fill(0.0)
        self.precomputation()
        lhs_matrix_np = self.lhs_matrix.to_numpy()
        s_lhs_matrix_np = sparse.csr_matrix(lhs_matrix_np)
        pre_fact_lhs_solve = factorized(s_lhs_matrix_np)
        gui = ti.GUI('Projective Dynamics Demo3 v0.2')
        gui.circles(self.pos.to_numpy(), radius=2, color=0xffaa33)
        if not os.path.exists("./results/"):
            os.mkdir("./results/")
        filename = f'./results/frame_rest.png'
        gui.show(filename)
        frame_counter = 0
        plot_array = []
        self.initial_com = self.calcCenterOfMass(np.arange(self.n_particles))  # this is right
        self.initial_rel_pos = self.pos_init.to_numpy() - self.initial_com
        while frame_counter < frame_count:
            self.build_sn()
            self.warm_up()  # Warm up:
            print("//////////////////////////////////////Frame ", frame_counter, "/////////////////////////////////")
            last_record_energy = 100000000000.0
            self.copy(self.pos, self.input_xn)
            # print("lhs_matrix_np: \n", lhs_matrix_np)
            self.copy(self.vel, self.input_vn)
            pn_v, pn_dis = pn.data_one_frame(self.input_xn, self.input_vn)
            for itr in range(self.solver_max_iteration):
                self.local_solve_build_bp_for_all_constraints()
                self.build_rhs(self.rhs_np)
                local_step_energy = self.compute_local_step_energy()
                # print("energy after local step:", local_step_energy)
                if local_step_energy > last_record_energy:
                    print("Energy Error: LOCAL; Error Amount:",
                          (local_step_energy - last_record_energy) / local_step_energy)
                    if (local_step_energy - last_record_energy) / local_step_energy > 0.01:
                        print("Large Error: LOCAL")
                last_record_energy = local_step_energy
                pos_new_np = pre_fact_lhs_solve(self.rhs_np)
                self.update_pos_new_from_numpy(pos_new_np)
                global_step_energy = self.compute_global_step_energy()
                # print("energy after global step:", global_step_energy)
                plot_array.append([itr, global_step_energy])
                if global_step_energy > last_record_energy:
                    print("Energy Error: GLOBAL; Error Amount:",
                          (global_step_energy - last_record_energy) / global_step_energy)
                    if (global_step_energy - last_record_energy) / global_step_energy > 0.01:
                        print("Large Error: GLOBAL")
                last_record_energy = global_step_energy
            # Update velocity and positions
            self.update_velocity_pos()
            self.paint_phi(gui)
            self.compute_x_xtilde()
            self.gradE, _ = pn.get_gradE_from_pd(self.pos)
            gui.circles(self.pos.to_numpy(), radius=2, color=0xd1d1d1)
            self.output_all(self.pos_delta2.to_numpy(), pn_dis.to_numpy(), self.gradE, frame_counter, is_test)
            frame_counter += 1
            filename = f'./results/frame_{frame_counter:05d}.png'
            gui.show(filename)


if __name__ == "__main__":
    pd = PDSimulation(int(1), 2)
    init_rel_pos = np.array(
        [[4.00000007e+00, -1.07179634e-08],
         [3.80422614e+00, 1.23606795e+00],
         [3.23606807e+00, 2.35114096e+00],
         [2.35114113e+00, 3.23606793e+00],
         [1.23606813e+00, 3.80422603e+00],
         [1.74850114e-07, 3.99999999e+00],
         [-1.23606779e+00, 3.80422609e+00],
         [-2.35114082e+00, 3.23606805e+00],
         [-3.23606781e+00, 2.35114114e+00],
         [-3.80422594e+00, 1.23606815e+00],
         [-3.99999993e+00, 2.03641209e-07],
         [-3.80422607e+00, -1.23606776e+00],
         [-3.23606806e+00, -2.35114081e+00],
         [-2.35114117e+00, -3.23606782e+00],
         [-1.23606820e+00, -3.80422598e+00],
         [-2.53868231e-07, -4.00000001e+00],
         [1.23606772e+00, -3.80422618e+00],
         [2.35114078e+00, -3.23606820e+00],
         [3.23606782e+00, -2.35114133e+00],
         [3.80422601e+00, -1.23606838e+00]])

    curr_rel_pos = np.array(
        [[4.00000007e+00, - 1.07179670e-08],
         [3.80422614e+00, 1.23606795e+00],
         [3.23606807e+00, 2.35114096e+00],
         [2.35114113e+00, 3.23606793e+00],
         [1.23606813e+00, 3.80422603e+00],
         [1.74850114e-07, 3.99999999e+00],
         [-1.23606779e+00, 3.80422609e+00],
         [-2.35114082e+00, 3.23606805e+00],
         [-3.23606781e+00, 2.35114114e+00],
         [-3.80422594e+00, 1.23606815e+00],
         [-3.99999993e+00, 2.03641205e-07],
         [-3.80422607e+00, - 1.23606776e+00],
         [-3.23606806e+00, - 2.35114081e+00],
         [-2.35114117e+00, - 3.23606782e+00],
         [-1.23606820e+00, - 3.80422598e+00],
         [-2.53868231e-07, - 4.00000001e+00],
         [1.23606772e+00, - 3.80422618e+00],
         [2.35114078e+00, - 3.23606820e+00],
         [3.23606782e+00, - 2.35114133e+00],
         [3.80422601e+00, - 1.23606838e+00]])

    pos = np.array(
        [[4.00000000e+00, 1.49972778e+01],
         [3.80422607e+00, 1.62333457e+01],
         [3.23606800e+00, 1.73484188e+01],
         [2.35114106e+00, 1.82333457e+01],
         [1.23606806e+00, 1.88015038e+01],
         [1.07179586e-07, 1.89972778e+01],
         [-1.23606786e+00, 1.88015039e+01],
         [-2.35114089e+00, 1.82333458e+01],
         [-3.23606788e+00, 1.73484189e+01],
         [-3.80422601e+00, 1.62333459e+01],
         [-4.00000000e+00, 1.49972780e+01],
         [-3.80422614e+00, 1.37612100e+01],
         [-3.23606813e+00, 1.26461370e+01],
         [-2.35114123e+00, 1.17612100e+01],
         [-1.23606826e+00, 1.11930518e+01],
         [-3.21538759e-07, 1.09972778e+01],
         [1.23606765e+00, 1.11930516e+01],
         [2.35114071e+00, 1.17612096e+01],
         [3.23606775e+00, 1.26461365e+01],
         [3.80422594e+00, 1.37612094e+01]])

    curr_com = np.array([-6.76705279e-08, 1.49972778e+01])
    A_pq = pd.calcA_pq(curr_rel_pos, init_rel_pos)
    print("Apq: \n", A_pq)
    R = calcR(A_pq)
    print("R: \n", R)
    RR = pd.calcR(A_pq)
    print("R: \n", RR)