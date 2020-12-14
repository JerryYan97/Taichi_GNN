import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.JGSL_WATER import *
import taichi as ti
import numpy as np
import pymesh
# from fixed_corotated import *
from Utils.neo_hookean import *
from Utils.math_tools import *
from Utils.reader import *
from .PD import*
from Utils.Dijkstra import *

from numpy.linalg import inv
from scipy.linalg import sqrtm
from numpy import linalg as LA

directory = ''

##############################################################################

real = ti.f64
scalar = lambda: ti.field(real)
vec = lambda: ti.Vector(2, dt=real)
mat = lambda: ti.Matrix(2, 2, dt=real)

@ti.data_oriented
class PNSimulation:
    def __init__(self, objfilenum, _dim):
        self.gui = ti.GUI("MPM", (1024, 1024), background_color=0x112F41)
        ################################ mesh ######################################
        self.mesh, self.dirichlet, self.mesh_scale, self.mesh_offset = read(objfilenum)

        ################################ material ######################################
        self.dim = _dim
        self.dt = 0.01
        self.E = 1e4
        self.nu = 0.4
        self.density = 1e2
        self.la = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))

        ################################ field ######################################
        self.n_particles = self.mesh.num_vertices
        self.n_elements = self.mesh.num_faces

        self.x = ti.Vector.field(self.dim, real, self.n_particles)
        self.xPrev = ti.Vector.field(self.dim, real, self.n_particles)
        self.xTilde = ti.Vector.field(self.dim, real, self.n_particles)
        self.xn = ti.Vector.field(self.dim, real, self.n_particles)
        self.v = ti.Vector.field(self.dim, real, self.n_particles)
        self.m = ti.field(real, self.n_particles)

        self.grad_x = ti.Vector.field(self.dim, real, self.n_particles)
        self.x_xtilde = ti.Vector.field(self.dim, real, self.n_particles)

        self.temp_x = ti.Vector.field(self.dim, real, self.n_particles)
        self.input_xn = ti.Vector.field(self.dim, real, self.n_particles)
        self.input_vn = ti.Vector.field(self.dim, real, self.n_particles)
        self.del_p = ti.Vector.field(self.dim, real, self.n_particles)

        self.F = ti.Matrix.field(self.dim, self.dim, real, self.n_elements)
        self.RR = ti.Matrix.field(self.dim, self.dim, real, self.n_particles)
        self.zero = ti.Vector.field(self.dim, real, self.n_particles)
        self.restT = ti.Matrix.field(self.dim, self.dim, real, self.n_elements)

        self.vertices = ti.field(ti.i32)

        ti.root.dense(ti.ij, (self.n_elements, 3)).place(self.vertices)

        self.data_rhs = ti.field(real, shape=2000)
        self.data_mat = ti.field(real, shape=(3, 100000))
        self.data_sol = ti.field(real, shape=2000)

        self.gradE = ti.field(real, shape=2000)
        self.cnt = ti.field(ti.i32, shape=())

        ################################ external force ######################################
        self.exf_ind = self.mag_ind = 0  # Used to label output .csv files.
        self.ex_force = ti.Vector.field(self.dim, real, 1)

        ################################ shape matching ######################################
        self.initial_com = ti.Vector([0.0, 0.0])
        self.initial_rel_pos = np.array([self.n_particles, 2])
        self.pi = ti.Vector.field(self.dim, real, self.n_particles)
        self.qi = ti.Vector.field(self.dim, real, self.n_particles)
        self.init_pos = self.mesh.vertices.astype(np.float64)[:, :2]

        ################################ initial ######################################
        self.x.from_numpy(self.mesh.vertices.astype(np.float64))
        self.vertices.from_numpy(self.mesh.faces)
        self.vertices_ = self.mesh.faces

    def set_material(self, _rho, _ym, _nu, _dt):
        self.dt = _dt
        self.density = _rho
        self.E = _ym
        self.nu = _nu
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.la = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def set_force(self, ind, mag):
        self.exf_ind = ind
        self.mag_ind = mag
        x = mag * ti.cos(3.1415926 / 180.0 * ind)
        y = mag * ti.sin(3.1415926 / 180.0 * ind)
        self.ex_force[0][0], self.ex_force[0][1] = x, y
        print("ex_force:", self.ex_force)


    @ti.func
    def compute_T(self, i):
        if ti.static(self.dim == 2):
            ab = self.x[self.vertices[i, 1]] - self.x[self.vertices[i, 0]]
            ac = self.x[self.vertices[i, 2]] - self.x[self.vertices[i, 0]]
            return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])

    @ti.func
    def compute_T_grad(self, i):
        if ti.static(self.dim == 2):
            ab = self.grad_x[self.vertices[i, 1]] - self.grad_x[self.vertices[i, 0]]
            ac = self.grad_x[self.vertices[i, 2]] - self.grad_x[self.vertices[i, 0]]
            return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])

    @ti.kernel
    def compute_restT_and_m(self):
        for i in range(self.n_elements):
            self.restT[i] = self.compute_T(i)
            # mass = self.restT[i].determinant() / 2 * self.density / 3
            mass = self.restT[i].determinant() / self.dim / (self.dim - 1) * self.density / (self.dim + 1)
            if mass < 0.0:
                print("FATAL ERROR : mesh inverted")
            for d in ti.static(range(3)):
                self.m[self.vertices[i, d]] += mass

    @ti.kernel
    def compute_xn_and_xTilde(self):
        for i in range(self.n_particles):
            self.xn[i] = self.x[i]
            self.xTilde[i] = self.x[i] + self.dt * self.v[i]
            self.xTilde(0)[i] += self.dt * self.dt * (self.ex_force[0][0]/self.m[i])
            self.xTilde(1)[i] += self.dt * self.dt * (self.ex_force[0][1]/self.m[i])

    @ti.kernel
    def compute_energy(self) -> real:
        total_energy = 0.0
        # inertia
        for i in range(self.n_particles):
            total_energy += 0.5 * self.m[i] * (self.x[i] - self.xTilde[i]).norm_sqr()
        # elasticity
        for e in range(self.n_elements):
            tempF = self.compute_T(e) @ self.restT[e].inverse()
            vol0 = self.restT[e].determinant() / 2
            U, sig, V = ti.svd(tempF)
            total_energy += fixed_corotated_energy(ti.Vector([sig[0, 0], sig[1, 1]]), self.la, self.mu) * self.dt * self.dt * vol0
        return total_energy

    @ti.kernel
    def compute_pd_gradient(self):
        for i in range(self.n_particles):
            for d in ti.static(range(2)):
                self.data_rhs[i * 2 + d] -= self.m[i] * (self.grad_x(d)[i] - self.xTilde(d)[i])
        for e in range(self.n_elements):
            self.F[e] = self.compute_T_grad(e) @ self.restT[e].inverse()
            IB = self.restT[e].inverse()
            vol0 = self.restT[e].determinant() / 2
            P = fixed_corotated_first_piola_kirchoff_stress(self.F[e], self.la, self.mu) * self.dt * self.dt * vol0
            self.data_rhs[self.vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
            self.data_rhs[self.vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
            self.data_rhs[self.vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
            self.data_rhs[self.vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
            self.data_rhs[self.vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
            self.data_rhs[self.vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]


    def get_gradE_from_pd(self, pd_pos):
        self.copy(pd_pos, self.grad_x)
        self.copy(pd_pos, self.x)
        self.compute_pd_gradient()
        self.copy(self.data_rhs, self.gradE)
        self.copy(pd_pos, self.x)
        self.compute_x_xtilde()
        return self.gradE, self.x_xtilde

    @ti.kernel
    def compute_hessian_and_gradient(self):
        self.cnt[None] = 0
        # inertia
        for i in range(self.n_particles):
            for d in ti.static(range(2)):
                c = self.cnt[None] + i * 2 + d
                self.data_mat[0, c] = i * 2 + d
                self.data_mat[1, c] = i * 2 + d
                self.data_mat[2, c] = self.m[i]
                self.data_rhs[i * 2 + d] -= self.m[i] * (self.x(d)[i] - self.xTilde(d)[i])
        self.cnt[None] += self.n_particles * 2
        # elasticity
        for e in range(self.n_elements):
            self.F[e] = self.compute_T(e) @ self.restT[e].inverse()
            IB = self.restT[e].inverse()
            vol0 = self.restT[e].determinant() / 2
            dPdF = fixed_corotated_first_piola_kirchoff_stress_derivative(self.F[e], self.la, self.mu) * self.dt * self.dt * vol0
            intermediate = ti.Matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
            for colI in ti.static(range(4)):
                _000 = dPdF[0, colI] * IB[0, 0]
                _010 = dPdF[0, colI] * IB[1, 0]
                _101 = dPdF[2, colI] * IB[0, 1]
                _111 = dPdF[2, colI] * IB[1, 1]
                _200 = dPdF[1, colI] * IB[0, 0]
                _210 = dPdF[1, colI] * IB[1, 0]
                _301 = dPdF[3, colI] * IB[0, 1]
                _311 = dPdF[3, colI] * IB[1, 1]
                intermediate[2, colI] = _000 + _101
                intermediate[3, colI] = _200 + _301
                intermediate[4, colI] = _010 + _111
                intermediate[5, colI] = _210 + _311
                intermediate[0, colI] = -intermediate[2, colI] - intermediate[4, colI]
                intermediate[1, colI] = -intermediate[3, colI] - intermediate[5, colI]
            indMap = ti.Vector([self.vertices[e, 0] * 2, self.vertices[e, 0] * 2 + 1,
                                self.vertices[e, 1] * 2, self.vertices[e, 1] * 2 + 1,
                                self.vertices[e, 2] * 2, self.vertices[e, 2] * 2 + 1])
            for colI in ti.static(range(6)):
                _000 = intermediate[colI, 0] * IB[0, 0]
                _010 = intermediate[colI, 0] * IB[1, 0]
                _101 = intermediate[colI, 2] * IB[0, 1]
                _111 = intermediate[colI, 2] * IB[1, 1]
                _200 = intermediate[colI, 1] * IB[0, 0]
                _210 = intermediate[colI, 1] * IB[1, 0]
                _301 = intermediate[colI, 3] * IB[0, 1]
                _311 = intermediate[colI, 3] * IB[1, 1]
                c = self.cnt[None] + e * 36 + colI * 6 + 0
                self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[2], indMap[colI], _000 + _101
                c = self.cnt[None] + e * 36 + colI * 6 + 1
                self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[3], indMap[colI], _200 + _301
                c = self.cnt[None] + e * 36 + colI * 6 + 2
                self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[4], indMap[colI], _010 + _111
                c = self.cnt[None] + e * 36 + colI * 6 + 3
                self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[5], indMap[colI], _210 + _311
                c = self.cnt[None] + e * 36 + colI * 6 + 4
                self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[0], indMap[colI], - _000 - _101 - _010 - _111
                c = self.cnt[None] + e * 36 + colI * 6 + 5
                self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[1], indMap[colI], - _200 - _301 - _210 - _311
            P = fixed_corotated_first_piola_kirchoff_stress(self.F[e], self.la, self.mu) * self.dt * self.dt * vol0
            self.data_rhs[self.vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
            self.data_rhs[self.vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
            self.data_rhs[self.vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
            self.data_rhs[self.vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
            self.data_rhs[self.vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
            self.data_rhs[self.vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]
        self.cnt[None] += self.n_elements * 36

    @ti.kernel
    def save_xPrev(self):
        for i in range(self.n_particles):
            self.xPrev[i] = self.x[i]

    @ti.kernel
    def apply_sol(self, alpha: real):
        for i in range(self.n_particles):
            for d in ti.static(range(2)):
                self.x(d)[i] = self.xPrev(d)[i] + self.data_sol[i * 2 + d] * alpha

    @ti.kernel
    def compute_x_xtilde(self):
        for i in range(self.n_particles):
            self.x_xtilde[i] = self.x[i] - self.xTilde[i]

    # @ti.kernel
    def copy_sol(self, sol):
        for i in range(self.n_particles):
            for d in ti.static(range(2)):
                self.data_sol[i*2+d] = sol[i*2+d]

    @ti.kernel
    def compute_v(self):
        for i in range(self.n_particles):
            self.v[i] = (self.x[i] - self.xn[i]) / self.dt
            self.del_p[i] = self.x[i] - self.xn[i]

    @ti.kernel
    def output_residual(self) -> real:
        residual = 0.0
        for i in range(self.n_particles):
            for d in ti.static(range(2)):
                residual = ti.max(residual, ti.abs(self.data_sol[i * 2 + d]))
        print("Search Direction Residual : ", residual / self.dt)
        return residual

    def write_image(self, f):
        particle_pos = (self.x.to_numpy() + self.mesh_offset) * self.mesh_scale
        for i in range(self.n_elements):
            for j in range(3):
                a, b = self.vertices_[i, j], self.vertices_[i, (j + 1) % 3]
                self.gui.line((particle_pos[a][0], particle_pos[a][1]),
                         (particle_pos[b][0], particle_pos[b][1]),
                         radius=1,
                         color=0x4FB99F)
        for i in self.dirichlet:
            self.gui.circle(particle_pos[i], radius=3, color=0x44FFFF)
        self.gui.show()

    # @ti.func
    def calcCenterOfMass(self, vind):
        sum = ti.Vector([0.0, 0.0])
        summ = 0.0
        for i in vind:
            for d in ti.static(range(2)):
                sum[d] += self.m[i] * self.x[i][d]
            summ += self.m[i]
        sum[0] /= summ
        sum[1] /= summ
        return sum

    @ti.func
    def calcA_qq(self, num_neigh):
        sum = ti.Matrix([[0, 0], [0, 0]])
        for i in range(num_neigh):
            sum += ti.outer_product(self.qi[i], self.qi[i].transpose())
        return sum.inverse()

    def calcA_pq(self, p_i, q_i):
        sum = np.zeros((2, 2))
        for i in range(p_i.shape[0]):
            sum += np.outer(p_i[i], np.transpose(q_i[i]))
        return sum

    # @ti.func
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
            t = t+1
        return result, result_pos

    # @ti.kernel
    def extreme_test(self):
        for i in range(11, self.n_particles):
            for d in ti.static(range(2)):
                pd = np.random.rand()
                self.x(d)[i] = 0.2 + (pd-0.5)

    # copy x to y
    @ti.kernel
    def copy(self, x: ti.template(), y: ti.template()):
        for i in x:
            y[i] = x[i]

    def output_all(self, pd_dis, pn_dis, grad_E, frame, T):
        frame = str(frame).zfill(2)
        if T == 0:
            out_name = "Outputs/output"+str(self.exf_ind)+"_"+str(self.mag_ind)+"_"+frame+".txt"
        else:
            out_name = "Outputs_T/output" + str(self.exf_ind) + "_" + str(self.mag_ind) + "_" + frame + ".txt"
        if not os.path.exists(out_name):
            file = open(out_name, 'w+')
            file.close()
        ele_count = self.dim + self.dim + self.dim * self.dim + self.dim + self.dim  # pd pos + pn pos + local transform + residual + force
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
            new_pos, init_rel_pos = self.build_pos_arr(adj_v, self.x, self.initial_rel_pos)
            com = self.calcCenterOfMass(adj_v).to_numpy()
            curr_rel_pos = np.zeros((adj_v.shape[0], self.dim))
            for j in range(new_pos.shape[0]):
                curr_rel_pos[j, :] = new_pos[j, :] - com
            A_pq = self.calcA_pq(curr_rel_pos, init_rel_pos)
            R = self.calcR(A_pq)
            out[i, 4] = R[0, 0]
            out[i, 5] = R[0, 1]
            out[i, 6] = R[1, 0]
            out[i, 7] = R[1, 1]
            out[i, 8] = grad_E[i*2]
            out[i, 9] = grad_E[i*2+1]
            # out[i, 10] = self.ex_force[0]
            # out[i, 11] = self.ex_force[1]
            out[i, 10] = self.ex_force[0][0]
            out[i, 11] = self.ex_force[0][1]
        np.savetxt(out_name, out)

    def data_one_frame(self, input_p, input_v):
        self.copy(input_p, self.x)
        self.copy(input_v, self.v)
        self.initial_com = self.calcCenterOfMass(np.arange(self.n_particles))
        self.initial_rel_pos = self.x.to_numpy() - self.initial_com
        # print("init center of mass: ", self.initial_com)
        self.compute_xn_and_xTilde()
        # print(self.dt, self.ex_force.to_numpy())
        # print(self.xTilde.to_numpy())
        while True:
            self.data_mat.fill(0)
            self.data_rhs.fill(0)
            self.data_sol.fill(0)
            self.compute_hessian_and_gradient()
            self.data_sol.from_numpy(
                solve_linear_system(self.data_mat.to_numpy(), self.data_rhs.to_numpy(), self.n_particles * self.dim,
                                    self.dirichlet, self.zero.to_numpy(), False, 0, self.cnt[None]))
            if self.output_residual() < 1e-4 * self.dt:
                break
            E0 = self.compute_energy()
            self.save_xPrev()
            alpha = 1.0
            self.apply_sol(alpha)
            E = self.compute_energy()
            while E > E0:
                alpha *= 0.5
                self.apply_sol(alpha)
                E = self.compute_energy()
        self.compute_v()
        return self.del_p, self.x, self.v

    def Run(self, is_test, frame_count):
        self.v.fill(0)
        self.m.fill(0)
        self.compute_restT_and_m()
        self.zero.fill(0)
        self.write_image(0)
        total_time = 0.0
        self.initial_com = self.calcCenterOfMass(np.arange(self.n_particles))
        self.initial_rel_pos = self.x.to_numpy() - self.initial_com

        for f in range(frame_count):
            total_time -= time.time()
            print("==================== Frame: ", f, " ====================")
            self.compute_xn_and_xTilde()
            # pn to solve
            while True:
                self.data_mat.fill(0)
                self.data_rhs.fill(0)
                self.data_sol.fill(0)
                self.compute_hessian_and_gradient()
                self.data_sol.from_numpy(solve_linear_system(self.data_mat.to_numpy(), self.data_rhs.to_numpy(),
                                                             self.n_particles * self.dim, self.dirichlet,
                                                             self.zero.to_numpy(), False, 0, self.cnt[None]))
                if self.output_residual() < 1e-4 * self.dt:
                    break
                E0 = self.compute_energy()
                self.save_xPrev()
                alpha = 1.0
                self.apply_sol(alpha)
                E = self.compute_energy()
                while E > E0:
                    alpha *= 0.5
                    self.apply_sol(alpha)
                    E = self.compute_energy()
                    print(alpha, E)
            # update
            self.compute_v()
            self.compute_x_xtilde()
            total_time += time.time()
            self.write_image(f + 1)

