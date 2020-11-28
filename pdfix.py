# Acknowledgement: ti example fem99.py
# Demo 3
import taichi as ti
import numpy as np
import sys, os, time
import csv
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from reader import*
import pymesh
import math

ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)
real = ti.f64


@ti.data_oriented
class PD_Simulation:
    def __init__(self, objfilenum, _dim):
        self.dim = 2
        self.dt = 0.01

        self.mesh, _, _, _ = read(int(objfilenum))
        self.NV, self.NF, _ = self.mesh.num_vertices, self.mesh.num_faces, self.mesh.num_voxels
        
        self.rho = 100
        self.E = 1e4
        self.nu = 0.4  # Young's modulus and Poisson's ratio
        self.mu = self.E / (2*(1+self.nu)) # Lame parameters
        self.lam = self.E * self.nu / ((1+self.nu)*(1-2*self.nu))

        self.ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.32

        # Area: 0.000061 0.02*0.02*sin90*0.5
        self.volume = ti.field(real, self.NF)
        self.m_weight_strain = ti.field(real, self.NF)  #  self.mu * 2 * self.volume
        self.m_weight_volume = ti.field(real, self.NF)  #  self.lam * self.dim * self.volume

        self.m_weight_positional = 10000000.0
        # self.print("m_weight_strain/volume", m_weight_strain/volume, "  m_weight_volume/volume", m_weight_volume/volume)

        self.mass = ti.field(ti.f64, self.NV)

        self.pos = ti.Vector.field(2, ti.f64, self.NV)
        self.pos_new = ti.Vector.field(2, ti.f64, self.NV)
        self.pos_init = ti.Vector.field(2, ti.f64, self.NV)
        self.last_pos_new = ti.Vector.field(2, ti.f64, self.NV)
        self.boundary_labels = ti.field(int, self.NV)

        self.pos_delta = ti.var(real, shape=2*self.NV)
        self.pos_delta2 = ti.Vector.field(2, real, self.NV)

        self.vel = ti.Vector.field(2, ti.f64, self.NV)
        self.f2v = ti.Vector.field(3, int, self.NF)         # ids of three vertices of each face
        self.B = ti.Matrix.field(2, 2, ti.f64, self.NF)     # The inverse of the init elements -- Dm
        self.F = ti.Matrix.field(2, 2, ti.f64, self.NF, needs_grad=True)
        self.A = ti.Matrix.field(4, 6, ti.f64, self.NF * 2)
        self.Bp = ti.Matrix.field(2, 2, ti.f64, self.NF * 2)
        
        self.rhs_np = np.zeros(self.NV * 2, dtype=np.float64)
        self.Sn = ti.field(ti.f64, self.NV * 2)

        self.lhs_matrix = ti.field(ti.f64, shape=(self.NV * 2, self.NV * 2))
        self.phi = ti.field(ti.f64, self.NF)  # potential energy of each element(face) for linear coratated elasticity material.

        self.resolutionX = 512
        self.pixels = ti.var(ti.f32, shape=(self.resolutionX, self.resolutionX))

        self.drag = 0.0  # 0.2

        self.solver_max_iteration = 10
        self.solver_stop_residual = 0.0001

         # force field
        self.gravity = ti.Vector([0, 0])
        self.ex_force = ti.Vector.field(self.dim, real, 1)

        # boundary setting
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


    def set_Material(self, _rho, _ym, _nu, _dt):
        self.dt = _dt
        self.rho = _rho
        self.E = _ym
        self.nu = _nu
        self.mu = self.E / (2*(1+self.nu))
        self.lam = self.E * self.nu / ((1+self.nu)*(1-2*self.nu))


    def set_Force(self, ex_force):
        print("ex_force:", ex_force)
        self.ex_force[0][0] = ex_force[0]
        self.ex_force[0][1] = ex_force[1]


    def init_mesh_obj(self):
        for i in range(self.mesh.num_faces):
            self.f2v[i] = ti.Vector([self.mesh.faces[i][0], self.mesh.faces[i][1], self.mesh.faces[i][2]])
        for i in range(self.mesh.num_vertices):
            self.pos[i] = ti.Vector([self.mesh.vertices[i][0], self.mesh.vertices[i][1]])  # + ti.Vector([0.2, 0.4]) # 0.2, 0.4 - 0.6,0.6  0.02*0.02
            self.pos_init[i] = ti.Vector([self.mesh.vertices[i][0], self.mesh.vertices[i][1]])
            self.vel[i] = ti.Vector([0, 0])
            if i in self.boundary_points_ and i <= 10:
                self.boundary_labels[i] = 1
            else:
                self.boundary_labels[i] = 0

    @ti.kernel
    def init_mesh_B(self):
        for i in range(self.NF): # NF number of face
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
            self.mass[ia] += self.volume[e_it]/dimp * self.rho
            self.mass[ib] += self.volume[e_it]/dimp * self.rho
            self.mass[ic] += self.volume[e_it]/dimp * self.rho

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
                self.A[t*self.NF+i][0, 0] = -a-c
                self.A[t*self.NF+i][0, 2] = a
                self.A[t*self.NF+i][0, 4] = c
                self.A[t*self.NF+i][1, 0] = -b-d
                self.A[t*self.NF+i][1, 2] = b
                self.A[t*self.NF+i][1, 4] = d
                self.A[t*self.NF+i][2, 1] = -a-c
                self.A[t*self.NF+i][2, 3] = a
                self.A[t*self.NF+i][2, 5] = c
                self.A[t*self.NF+i][3, 1] = -b-d
                self.A[t*self.NF+i][3, 3] = b
                self.A[t*self.NF+i][3, 5] = d

        # Construct lhs matrix without constraints
        for i in range(self.NV):
            for d in ti.static(range(2)):
                self.lhs_matrix[i*self.dim+d, i*self.dim+d] += (self.drag/self.dt)+ self.mass[i]/(self.dt*self.dt)

        # Add strain and area/volume constraints to the lhs matrix
        for t in ti.static(range(2)):
            for ele_idx in range(self.NF):
                A_i = self.A[t*self.NF+ele_idx]
                ia, ib, ic = self.f2v[ele_idx]
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
                                weight = self.m_weight_strain[ele_idx]
                            else:
                                weight = self.m_weight_volume[ele_idx]
                            self.lhs_matrix[lhs_row_idx, lhs_col_idx] += (A_i[idx, A_row_idx]*A_i[idx, A_col_idx]*weight)

        # Add positional constraints to the lhs matrix
        for i in range(self.NV):
            if self.boundary_labels[i] == 1:
                q_i_x_idx = i * 2
                q_i_y_idx = i * 2 + 1
                self.lhs_matrix[q_i_x_idx, q_i_x_idx] += self.m_weight_positional  # This is the weight of positional constraints
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
            energy1 = self.mu*self.volume[i]*((F_i - Bp_i_strain).norm()**2)
            energy2 = 0.5*self.lam*self.volume[i]*((F_i - Bp_i_volume).trace()**2)
            self.phi[i] = energy1 + energy2


    @ti.kernel
    def build_sn(self):
        for vert_idx in range(self.NV):  # number of vertices
            Sn_idx1 = vert_idx*2  # m_sn
            Sn_idx2 = vert_idx*2+1
            pos_i = self.pos[vert_idx]  # pos = m_x
            vel_i = self.vel[vert_idx]
            self.Sn[Sn_idx1] = pos_i[0]+self.dt*vel_i[0]+(self.dt*self.dt)*(self.ex_force[0][0]/self.mass[vert_idx])  # x-direction;
            self.Sn[Sn_idx2] = pos_i[1]+self.dt*vel_i[1]+(self.dt*self.dt)*(self.ex_force[0][1]/self.mass[vert_idx])  # y-direction;


    @ti.kernel
    def build_rhs(self, rhs: ti.ext_arr()):
        one_over_dt2 = 1.0 / (self.dt ** 2)
        # Construct the first part of the rhs
        for i in range(self.NV * 2):
            pos_i = self.pos[i/2]
            p0 = pos_i[0]
            p1 = pos_i[1]
            if i % 2 == 0:
                rhs[i] = one_over_dt2 * self.mass[i/2] * self.Sn[i] + (self.drag/self.dt*p0)  # 0.000061
            else:
                rhs[i] = one_over_dt2 * self.mass[i/2] * self.Sn[i] + (self.drag/self.dt*p1)  # 0.000061
        # Add strain and volume/area constraints to the rhs
        for t in ti.static(range(2)):
            for ele_idx in range(self.NF):
                ia, ib, ic = self.f2v[ele_idx]
                Bp_i = self.Bp[t*self.NF+ele_idx]  # It is a 2x2 matrix now. We want it be a 4x1 vector.
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
            self.vel[i] = (self.pos_new[i] - self.pos[i]) / self.dt
            self.pos_delta2[i] = self.pos_new[i] - self.pos[i]
            self.pos[i] = self.pos_new[i]


    @ti.kernel
    def update_delta_pos(self):
        for i in range(self.NV):
            if self.boundary_labels[i] == 0:
                self.pos_delta[i] = self.pos_new[i][0] - self.pos[i][0]
                self.pos_delta[i+1] = self.pos_new[i][1] - self.pos[i][1]
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
            sol_idx1, sol_idx2 = pos_idx*2, pos_idx*2+1
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
            T1 += (temp_diff[0]**2 + temp_diff[1]**2)
        return T1 / (2.0 * self.dt**2)


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
        # Calculate T2 energy
        local_T2_energy = ti.cast(0.0, ti.f64)
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
        # Calculate global T2 energy
        global_T2_energy = self.global_compute_T2_energy()
        # Calculate global T1 energy
        global_T1_energy = self.compute_T1_energy()
        return (global_T1_energy + global_T2_energy)


    def compute_local_step_energy(self):
        local_T2_energy = self.local_compute_T2_energy()
        # Calculate T1 energy
        local_T1_energy = self.compute_T1_energy()
        return (local_T1_energy + local_T2_energy)


    def paint_phi(self, gui):
        pos_np = self.pos.to_numpy()
        phi_np = self.phi.to_numpy()
        f2v_np = self.f2v.to_numpy()
        a, b, c = pos_np[f2v_np[:, 0]], pos_np[f2v_np[:, 1]], pos_np[f2v_np[:, 2]]
        k = phi_np * (8000 / self.E)
        gb = (1 - k) * 0.7
        # print("gb:", gb[0])
        # print("phi_np", phi_np[0])
        # print("k", k[0])
        gui.triangles(a, b, c, color=ti.rgb_to_hex([k + gb, gb, gb]))
        gui.lines(a, b, color=0xffffff, radius=0.5)
        gui.lines(b, c, color=0xffffff, radius=0.5)
        gui.lines(c, a, color=0xffffff, radius=0.5)


    @ti.kernel
    def copy(self, x: ti.template(), y: ti.template()):
         for i in x:
            y[i] = x[i]


    def output_pos(self, np_pos, i):
        outname = "Output/output" + str(i) + ".txt"
        if not os.path.exists(outname):
            os.system(r"touch {}".format(outname))
        np.savetxt(outname, np_pos)


    def run(self):
        frame_counter = 0
        self.init_mesh_obj()
        self.init_mesh_B()
        self.precomputation()
        lhs_matrix_np = self.lhs_matrix.to_numpy()
        s_lhs_matrix_np = sparse.csr_matrix(lhs_matrix_np)
        pre_fact_lhs_solve = factorized(s_lhs_matrix_np)
        print("sparse lhs matrix:\n", s_lhs_matrix_np)
        self.initinfo()
        gui = ti.GUI('Projective Dynamics Demo3 v0.2')
        wait = input("PRESS ENTER TO CONTINUE.")
        gui.circles(self.pos.to_numpy(), radius=2, color=0xffaa33)
        filename = f'./results/frame_rest.png'
        gui.show(filename)

        frame_counter = 0
        sim_t = 0.0
        plot_array = []

        while frame_counter < 50:
            # print(self.pos.to_numpy())
            # print("////////////////////////////////////////////////////////////")
            # print(self.vel.to_numpy())
            self.build_sn()
            # Warm up:
            self.warm_up()
            print("Frame ", frame_counter)
            last_record_energy = 100000000000.0
            for itr in range(self.solver_max_iteration):
                self.local_solve_build_bp_for_all_constraints()
                self.build_rhs(self.rhs_np)
                local_step_energy = self.compute_local_step_energy()
                # print("energy after local step:", local_step_energy)
                if local_step_energy > last_record_energy:
                    print("Energy Error: LOCAL; Error Amount:", (local_step_energy - last_record_energy) / local_step_energy)
                    if (local_step_energy - last_record_energy) / local_step_energy > 0.01:
                        print("Large Error: LOCAL")
                last_record_energy = local_step_energy
                pos_new_np = pre_fact_lhs_solve(self.rhs_np)
                self.update_pos_new_from_numpy(pos_new_np)
                global_step_energy = self.compute_global_step_energy()
                # print("energy after global step:", global_step_energy)
                plot_array.append([itr, global_step_energy])
                if global_step_energy > last_record_energy:
                    print("Energy Error: GLOBAL; Error Amount:", (global_step_energy - last_record_energy) / global_step_energy)
                    if (global_step_energy - last_record_energy) / global_step_energy > 0.01:
                        print("Large Error: GLOBAL")
                last_record_energy = global_step_energy

            # Update velocity and positions
            self.update_velocity_pos()
            self.paint_phi(gui)
            # print(self.pos.to_numpy())
            gui.circles(self.pos.to_numpy(), radius=2, color=0xd1d1d1)
            frame_counter += 1
            filename = f'./results/frame_{frame_counter:05d}.png'
            gui.show(filename)
            # print("\n")
    
    
    def dataoneframe(self, input_p, input_v):
        lhs_matrix_np = self.lhs_matrix.to_numpy()
        s_lhs_matrix_np = sparse.csr_matrix(lhs_matrix_np)
        pre_fact_lhs_solve = factorized(s_lhs_matrix_np)
        self.copy(input_p, self.pos)
        self.copy(input_v, self.vel)
        # print(self.pos.to_numpy())
        print("////////////////////////////////////////////////////////////")
        # print(self.vel.to_numpy())
        # print(input_v.to_numpy())

        self.build_sn()
        self.warm_up()  # Warm up:
        last_record_energy = 100000000000.0
        for itr in range(self.solver_max_iteration):
            self.local_solve_build_bp_for_all_constraints()
            self.build_rhs(self.rhs_np)
            local_step_energy = self.compute_local_step_energy()
            if local_step_energy > last_record_energy:
                print("Energy Error: LOCAL; Error Amount:", (local_step_energy - last_record_energy) / local_step_energy)
                if (local_step_energy - last_record_energy) / local_step_energy > 0.01:
                    print("Large Error: LOCAL")
            last_record_energy = local_step_energy
            pos_new_np = pre_fact_lhs_solve(self.rhs_np)
            self.update_pos_new_from_numpy(pos_new_np)
            global_step_energy = self.compute_global_step_energy()
            if global_step_energy > last_record_energy:
                print("Energy Error: GLOBAL; Error Amount:", (global_step_energy - last_record_energy) / global_step_energy)
                if (global_step_energy - last_record_energy) / global_step_energy > 0.01:
                    print("Large Error: GLOBAL")
            last_record_energy = global_step_energy
        # Update velocity and positions
        self.update_velocity_pos()
        # self.update_delta_pos()
        return self.pos_delta2, self.pos_new, self.vel


    def init(self):
        self.init_mesh_obj()
        self.init_mesh_B()  # this is only for the rest position, so it is ok!
        self.precomputation()

# Energy Error note (under first 150 frames):
# 5 fixed iterations: 76; Local: 0; Global: 76; (5%)
# 10 fixed iterations: 421; Local: 0; Global: 420; (14%)
# 100 fixed iterations: 6584; Local: 2931; 3654; (21%)


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

if __name__ == "__main__":
    pd = PD_Simulation(int(sys.argv[1]), 2)
    pd.set_Force(13, 22)
    pd.run()