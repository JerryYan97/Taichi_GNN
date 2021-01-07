import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
print(sys.path)
from Utils.JGSL_WATER import *
import taichi as ti
import numpy as np
from Utils.neo_hookean import fixed_corotated_first_piola_kirchoff_stress
from Utils.neo_hookean import fixed_corotated_energy
from Utils.neo_hookean import fixed_corotated_first_piola_kirchoff_stress_derivative
from Utils.math_tools import svd
from Utils.Dijkstra import *
from numpy.linalg import inv
from scipy.linalg import sqrtm
from Utils.utils_visualization import draw_image, set_3D_scene, update_mesh, get_force_field, get_ring_force_field
##############################################################################
real = ti.f64


@ti.data_oriented
class PNSimulation:
    def __init__(self, case_info, _dt):
        ################################ mesh ######################################
        self.case_info = case_info
        self.mesh = self.case_info['mesh']
        self.dirichlet = self.case_info['dirichlet']
        self.mesh_scale = self.case_info['mesh_scale']
        self.mesh_offset = self.case_info['mesh_offset']
        self.dim = self.case_info['dim']
        self.center = self.case_info['center']
        self.min_sphere_radius = self.case_info['min_sphere_radius']
        ################################ material ######################################
        self.dt = _dt
        self.E = 1e4
        self.nu = 0.4
        self.density = 1e2
        self.la = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        ################################ field ######################################
        self.n_vertices = self.mesh.num_vertices
        if self.dim == 2:
            self.n_elements = self.mesh.num_faces
        if self.dim == 3:
            self.n_elements = self.mesh.num_elements
        # print("element count: ", self.n_elements)

        self.x = ti.Vector.field(self.dim, real, self.n_vertices)
        self.xPrev = ti.Vector.field(self.dim, real, self.n_vertices)
        self.xTilde = ti.Vector.field(self.dim, real, self.n_vertices)
        self.xn = ti.Vector.field(self.dim, real, self.n_vertices)
        self.v = ti.Vector.field(self.dim, real, self.n_vertices)
        self.m = ti.field(real, self.n_vertices)

        self.grad_x = ti.Vector.field(self.dim, real, self.n_vertices)
        self.x_xtilde = ti.Vector.field(self.dim, real, self.n_vertices)

        self.input_xn = ti.Vector.field(self.dim, real, self.n_vertices)
        self.input_vn = ti.Vector.field(self.dim, real, self.n_vertices)
        self.del_p = ti.Vector.field(self.dim, real, self.n_vertices)

        self.F = ti.Matrix.field(self.dim, self.dim, real, self.n_elements)
        self.zero = ti.Vector.field(self.dim, real, self.n_vertices)
        self.restT = ti.Matrix.field(self.dim, self.dim, real, self.n_elements)
        self.vertices = ti.field(ti.i32, (self.n_elements, self.dim + 1))

        self.data_rhs = np.zeros(shape=(200000,), dtype=np.float64)
        self.data_mat = np.zeros(shape=(3, 20000000), dtype=np.float64)
        self.data_sol = np.zeros(shape=(200000,), dtype=np.float64)

        self.gradE = ti.field(real, shape=200000)
        self.cnt = ti.field(ti.i32, shape=())

        # Complie time optimization data structure
        self.ti_dPdF_field = ti.field(real, (self.n_elements, self.dim * self.dim, self.dim * self.dim))
        self.ti_intermediate_field = ti.field(real, (self.n_elements, self.dim * (self.dim + 1), self.dim * self.dim))
        self.ti_M_field = ti.field(real, (self.n_elements, self.dim * self.dim, self.dim * self.dim))
        self.ti_U_field = ti.field(real, (self.n_elements, self.dim * self.dim, self.dim * self.dim))
        self.ti_V_field = ti.field(real, (self.n_elements, self.dim * self.dim, self.dim * self.dim))
        self.ti_indMap_field = ti.field(real, (self.n_elements, self.dim * (self.dim + 1)))
        ################################ external force ######################################
        self.exf_ind = self.mag_ind = 0  # Used to label output .csv files.
        self.ex_force = ti.Vector.field(self.dim, real, self.n_vertices)
        ################################ shape matching ######################################
        if self.dim == 2:
            self.initial_com = ti.Vector([0.0, 0.0])
        else:
            self.initial_com = ti.Vector([0.0, 0.0, 0.0])
        self.initial_rel_pos = np.array([self.n_vertices, self.dim])
        self.pi = ti.Vector.field(self.dim, real, self.n_vertices)
        self.qi = ti.Vector.field(self.dim, real, self.n_vertices)
        self.init_pos = self.mesh.vertices.astype(np.float32)[:, :self.dim]

        self.ti_center = []
        self.exf_name = ""
        self.ring_mag = 1.0
        self.ring_angle = 0.0
        self.ring_width = 0.2
        self.ring_circle_radius = 0.2
        self.dir_mag = 1.0
        self.dir_ang1 = 45.0
        self.dir_ang2 = 45.0

    def initial(self):
        self.x.from_numpy(self.mesh.vertices.astype(np.float64))
        if self.dim == 2:
            self.vertices.from_numpy(self.mesh.faces)
            self.vertices_ = self.mesh.faces
        if self.dim == 3:
            self.vertices.from_numpy(self.mesh.elements)
            self.vertices_ = self.mesh.elements

        self.xPrev.fill(0)
        self.xTilde.fill(0)
        self.xn.fill(0)
        self.v.fill(0)
        self.m.fill(0)

        self.grad_x.fill(0)
        self.x_xtilde.fill(0)

        self.input_xn.fill(0)
        self.input_vn.fill(0)
        self.del_p.fill(0)

        # self.F.fill(0)
        self.zero.fill(0)
        self.restT.fill(0)
        ################################ external force ######################################
        self.ex_force.fill(0)
        self.ti_center = ti.Vector([self.center[0], self.center[1], self.center[2]])

    def set_material(self, _rho, _ym, _nu, _dt):
        self.dt = _dt
        self.density = _rho
        self.E = _ym
        self.nu = _nu
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.la = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        # print("mu: ", self.mu, "la: ", self.la)

    def set_force(self, force_info):
        if force_info['dim'] != self.case_info['dim']:
            raise AttributeError("Input force dim is not equal to the simulator's dim!")
        if force_info['dim'] == 2:
            exf_angle = force_info['exf_angle']
            exf_mag = force_info['exf_mag']
            self.ex_force[0] = ti.Vector(get_force_field(exf_mag, exf_angle))
        else:
            exf_angle1 = force_info['exf_angle1']
            exf_angle2 = force_info['exf_angle2']
            exf_mag = force_info['exf_mag']
            self.ex_force[0] = ti.Vector(get_force_field(exf_mag, exf_angle1, exf_angle2, 3))

    @ti.kernel
    def set_ring_force_3D(self):
        for i in range(self.n_vertices):
            self.ex_force[i] = get_ring_force_field(self.ring_mag, self.ring_width,
                                                    self.ti_center, self.x[i],
                                                    self.ring_angle, 3)

    @ti.kernel
    def set_ring_circle_force_3D(self):
        for i in range(self.n_vertices):
            self.ex_force[i] = get_ring_force_field(self.ring_mag, self.ring_width,
                                                    self.ti_center, self.x[i], self.ring_angle,
                                                    self.ring_circle_radius * self.min_sphere_radius, 3)

    @ti.func
    def compute_T(self, i):
        if ti.static(self.dim == 2):
            ab = self.x[self.vertices[i, 1]] - self.x[self.vertices[i, 0]]
            ac = self.x[self.vertices[i, 2]] - self.x[self.vertices[i, 0]]
            return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])
        else:
            ab = self.x[self.vertices[i, 1]] - self.x[self.vertices[i, 0]]
            ac = self.x[self.vertices[i, 2]] - self.x[self.vertices[i, 0]]
            ad = self.x[self.vertices[i, 3]] - self.x[self.vertices[i, 0]]
            return ti.Matrix([[ab[0], ac[0], ad[0]], [ab[1], ac[1], ad[1]], [ab[2], ac[2], ad[2]]])

    @ti.func
    def compute_T_grad(self, i):
        if ti.static(self.dim == 2):
            ab = self.grad_x[self.vertices[i, 1]] - self.grad_x[self.vertices[i, 0]]
            ac = self.grad_x[self.vertices[i, 2]] - self.grad_x[self.vertices[i, 0]]
            return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])
        else:
            ab = self.grad_x[self.vertices[i, 1]] - self.grad_x[self.vertices[i, 0]]
            ac = self.grad_x[self.vertices[i, 2]] - self.grad_x[self.vertices[i, 0]]
            ad = self.grad_x[self.vertices[i, 3]] - self.grad_x[self.vertices[i, 0]]
            return ti.Matrix([[ab[0], ac[0], ad[0]], [ab[1], ac[1], ad[1]], [ab[2], ac[2], ad[2]]])

    @ti.kernel
    def compute_restT_and_m(self):
        for i in range(self.n_elements):
            self.restT[i] = self.compute_T(i)
            mass = self.restT[i].determinant() / self.dim / (self.dim - 1) * self.density / (self.dim + 1)
            if mass < 0.0:
                print("FATAL ERROR : mesh inverted")
            for d in ti.static(range(self.dim + 1)):
                self.m[self.vertices[i, d]] += mass

    @ti.kernel
    def compute_xn_and_xTilde(self):
        if ti.static(self.dim == 2):
            for i in range(self.n_vertices):
                self.xn[i] = self.x[i]
                self.xTilde[i] = self.x[i] + self.dt * self.v[i]
                self.xTilde(0)[i] += self.dt * self.dt * (self.ex_force[i][0]/self.m[i])
                self.xTilde(1)[i] += self.dt * self.dt * (self.ex_force[i][1]/self.m[i])
        if ti.static(self.dim == 3):
            for i in range(self.n_vertices):
                self.xn[i] = self.x[i]
                self.xTilde[i] = self.x[i] + self.dt * self.v[i]
                self.xTilde(0)[i] += self.dt * self.dt * (self.ex_force[i][0] / self.m[i])
                self.xTilde(1)[i] += self.dt * self.dt * (self.ex_force[i][1] / self.m[i])
                self.xTilde(2)[i] += self.dt * self.dt * (self.ex_force[i][2] / self.m[i])

    @ti.kernel
    def compute_energy(self) -> real:
        total_energy = 0.0
        # inertia
        for i in range(self.n_vertices):
            total_energy += 0.5 * self.m[i] * (self.x[i] - self.xTilde[i]).norm_sqr()
        # elasticity
        for e in range(self.n_elements):
            F = self.compute_T(e) @ self.restT[e].inverse()
            vol0 = self.restT[e].determinant() / self.dim / (self.dim - 1)
            U, sig, V = svd(F)
            total_energy += fixed_corotated_energy(sig, self.la, self.mu) * self.dt * self.dt * vol0
        return total_energy

    @ti.kernel
    def compute_pd_gradient(self, data_rhs_np: ti.ext_arr()):
        if ti.static(self.dim == 2):
            for i in range(self.n_vertices):
                for d in ti.static(range(self.dim)):
                    data_rhs_np[i * self.dim + d] -= self.m[i] * (self.grad_x(d)[i] - self.xTilde(d)[i])
            for e in range(self.n_elements):
                F = self.compute_T_grad(e) @ self.restT[e].inverse()
                IB = self.restT[e].inverse()
                vol0 = self.restT[e].determinant() / 2
                P = fixed_corotated_first_piola_kirchoff_stress(F, self.la, self.mu) * self.dt * self.dt * vol0
                data_rhs_np[self.vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
                data_rhs_np[self.vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
                data_rhs_np[self.vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
                data_rhs_np[self.vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
                data_rhs_np[self.vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
                data_rhs_np[self.vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]
        else:
            for i in range(self.n_vertices):
                for d in ti.static(range(self.dim)):
                    data_rhs_np[i * self.dim + d] -= self.m[i] * (self.grad_x(d)[i] - self.xTilde(d)[i])
            for e in range(self.n_elements):
                F = self.compute_T_grad(e) @ self.restT[e].inverse()
                IB = self.restT[e].inverse()
                vol0 = self.restT[e].determinant() / self.dim / (self.dim - 1)
                P = fixed_corotated_first_piola_kirchoff_stress(F, self.la, self.mu) * self.dt * self.dt * vol0
                R10 = IB[0, 0] * P[0, 0] + IB[0, 1] * P[0, 1] + IB[0, 2] * P[0, 2]
                R11 = IB[0, 0] * P[1, 0] + IB[0, 1] * P[1, 1] + IB[0, 2] * P[1, 2]
                R12 = IB[0, 0] * P[2, 0] + IB[0, 1] * P[2, 1] + IB[0, 2] * P[2, 2]
                R20 = IB[1, 0] * P[0, 0] + IB[1, 1] * P[0, 1] + IB[1, 2] * P[0, 2]
                R21 = IB[1, 0] * P[1, 0] + IB[1, 1] * P[1, 1] + IB[1, 2] * P[1, 2]
                R22 = IB[1, 0] * P[2, 0] + IB[1, 1] * P[2, 1] + IB[1, 2] * P[2, 2]
                R30 = IB[2, 0] * P[0, 0] + IB[2, 1] * P[0, 1] + IB[2, 2] * P[0, 2]
                R31 = IB[2, 0] * P[1, 0] + IB[2, 1] * P[1, 1] + IB[2, 2] * P[1, 2]
                R32 = IB[2, 0] * P[2, 0] + IB[2, 1] * P[2, 1] + IB[2, 2] * P[2, 2]
                data_rhs_np[self.vertices[e, 1] * 3 + 0] -= R10
                data_rhs_np[self.vertices[e, 1] * 3 + 1] -= R11
                data_rhs_np[self.vertices[e, 1] * 3 + 2] -= R12
                data_rhs_np[self.vertices[e, 2] * 3 + 0] -= R20
                data_rhs_np[self.vertices[e, 2] * 3 + 1] -= R21
                data_rhs_np[self.vertices[e, 2] * 3 + 2] -= R22
                data_rhs_np[self.vertices[e, 3] * 3 + 0] -= R30
                data_rhs_np[self.vertices[e, 3] * 3 + 1] -= R31
                data_rhs_np[self.vertices[e, 3] * 3 + 2] -= R32
                data_rhs_np[self.vertices[e, 0] * 3 + 0] -= -R10 - R20 - R30
                data_rhs_np[self.vertices[e, 0] * 3 + 1] -= -R11 - R21 - R31
                data_rhs_np[self.vertices[e, 0] * 3 + 2] -= -R12 - R22 - R32

    def get_gradE_from_pd(self, pd_pos):
        self.copy(pd_pos, self.grad_x)
        # self.copy(pd_pos, self.x)
        self.compute_pd_gradient(self.data_rhs)
        self.gradE.from_numpy(self.data_rhs)
        # self.copy(self.data_rhs, self.gradE)
        # self.copy(pd_pos, self.x)
        # self.compute_x_xtilde()
        return self.gradE

    @ti.kernel
    def compute_hessian_and_gradient(self, data_mat_np: ti.ext_arr(), data_rhs_np: ti.ext_arr()):
        self.cnt[None] = 0
        # inertia
        for i in range(self.n_vertices):
            for d in ti.static(range(self.dim)):
                c = self.cnt[None] + i * self.dim + d  # 2 * n_p
                # self.data_mat[0, c] = i * self.dim + d
                data_mat_np[0, c] = i * self.dim + d
                data_mat_np[1, c] = i * self.dim + d
                data_mat_np[2, c] = self.m[i]
                data_rhs_np[i * self.dim + d] -= self.m[i] * (self.x(d)[i] - self.xTilde(d)[i])
        self.cnt[None] += self.n_vertices * self.dim
        # elasticity
        for e in range(self.n_elements):
            F = self.compute_T(e) @ self.restT[e].inverse()  # 2 * 2
            IB = self.restT[e].inverse()  # 2 * 2
            vol0 = self.restT[e].determinant() / self.dim / (self.dim - 1)
            fixed_corotated_first_piola_kirchoff_stress_derivative(F, self.la, self.mu,
                                                                   self.ti_dPdF_field, self.ti_M_field, self.ti_U_field,
                                                                   self.ti_V_field, e, self.dt, vol0)
            P = fixed_corotated_first_piola_kirchoff_stress(F, self.la, self.mu) * self.dt * self.dt * vol0
            if ti.static(self.dim == 2):
                for colI in range(4):
                    _000 = self.ti_dPdF_field[e, 0, colI] * IB[0, 0]
                    _010 = self.ti_dPdF_field[e, 0, colI] * IB[1, 0]
                    _101 = self.ti_dPdF_field[e, 2, colI] * IB[0, 1]
                    _111 = self.ti_dPdF_field[e, 2, colI] * IB[1, 1]
                    _200 = self.ti_dPdF_field[e, 1, colI] * IB[0, 0]
                    _210 = self.ti_dPdF_field[e, 1, colI] * IB[1, 0]
                    _301 = self.ti_dPdF_field[e, 3, colI] * IB[0, 1]
                    _311 = self.ti_dPdF_field[e, 3, colI] * IB[1, 1]
                    self.ti_intermediate_field[e, 2, colI] = _000 + _101
                    self.ti_intermediate_field[e, 3, colI] = _200 + _301
                    self.ti_intermediate_field[e, 4, colI] = _010 + _111
                    self.ti_intermediate_field[e, 5, colI] = _210 + _311
                    self.ti_intermediate_field[e, 0, colI] = -self.ti_intermediate_field[e, 2, colI] - self.ti_intermediate_field[
                        e, 4, colI]
                    self.ti_intermediate_field[e, 1, colI] = -self.ti_intermediate_field[e, 3, colI] - self.ti_intermediate_field[
                        e, 5, colI]

                self.ti_indMap_field[e, 0] = self.vertices[e, 0] * 2
                self.ti_indMap_field[e, 1] = self.vertices[e, 0] * 2 + 1
                self.ti_indMap_field[e, 2] = self.vertices[e, 1] * 2
                self.ti_indMap_field[e, 3] = self.vertices[e, 1] * 2 + 1
                self.ti_indMap_field[e, 4] = self.vertices[e, 2] * 2
                self.ti_indMap_field[e, 5] = self.vertices[e, 2] * 2 + 1

                for colI in range(6):
                    _000 = self.ti_intermediate_field[e, colI, 0] * IB[0, 0]
                    _010 = self.ti_intermediate_field[e, colI, 0] * IB[1, 0]
                    _101 = self.ti_intermediate_field[e, colI, 2] * IB[0, 1]
                    _111 = self.ti_intermediate_field[e, colI, 2] * IB[1, 1]
                    _200 = self.ti_intermediate_field[e, colI, 1] * IB[0, 0]
                    _210 = self.ti_intermediate_field[e, colI, 1] * IB[1, 0]
                    _301 = self.ti_intermediate_field[e, colI, 3] * IB[0, 1]
                    _311 = self.ti_intermediate_field[e, colI, 3] * IB[1, 1]
                    c = self.cnt[None] + e * 36 + colI * 6 + 0
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 2], self.ti_indMap_field[
                        e, colI], _000 + _101
                    c = self.cnt[None] + e * 36 + colI * 6 + 1
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 3], self.ti_indMap_field[
                        e, colI], _200 + _301
                    c = self.cnt[None] + e * 36 + colI * 6 + 2
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 4], self.ti_indMap_field[
                        e, colI], _010 + _111
                    c = self.cnt[None] + e * 36 + colI * 6 + 3
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 5], self.ti_indMap_field[
                        e, colI], _210 + _311
                    c = self.cnt[None] + e * 36 + colI * 6 + 4
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 0], self.ti_indMap_field[
                        e, colI], - _000 - _101 - _010 - _111
                    c = self.cnt[None] + e * 36 + colI * 6 + 5
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 1], self.ti_indMap_field[
                        e, colI], - _200 - _301 - _210 - _311
                data_rhs_np[self.vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
                data_rhs_np[self.vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
                data_rhs_np[self.vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
                data_rhs_np[self.vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
                data_rhs_np[self.vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[
                    0, 1] * IB[1, 1]
                data_rhs_np[self.vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[
                    1, 1] * IB[1, 1]
            else:
                for colI in range(9):
                    self.ti_intermediate_field[e, 3, colI] = IB[0, 0] * self.ti_dPdF_field[e, 0, colI] + IB[0, 1] * self.ti_dPdF_field[
                        e, 3, colI] + IB[0, 2] * self.ti_dPdF_field[e, 6, colI]
                    self.ti_intermediate_field[e, 4, colI] = IB[0, 0] * self.ti_dPdF_field[e, 1, colI] + IB[0, 1] * self.ti_dPdF_field[
                        e, 4, colI] + IB[0, 2] * self.ti_dPdF_field[e, 7, colI]
                    self.ti_intermediate_field[e, 5, colI] = IB[0, 0] * self.ti_dPdF_field[e, 2, colI] + IB[0, 1] * self.ti_dPdF_field[
                        e, 5, colI] + IB[0, 2] * self.ti_dPdF_field[e, 8, colI]
                    self.ti_intermediate_field[e, 6, colI] = IB[1, 0] * self.ti_dPdF_field[e, 0, colI] + IB[1, 1] * self.ti_dPdF_field[
                        e, 3, colI] + IB[1, 2] * self.ti_dPdF_field[e, 6, colI]
                    self.ti_intermediate_field[e, 7, colI] = IB[1, 0] * self.ti_dPdF_field[e, 1, colI] + IB[1, 1] * self.ti_dPdF_field[
                        e, 4, colI] + IB[1, 2] * self.ti_dPdF_field[e, 7, colI]
                    self.ti_intermediate_field[e, 8, colI] = IB[1, 0] * self.ti_dPdF_field[e, 2, colI] + IB[1, 1] * self.ti_dPdF_field[
                        e, 5, colI] + IB[1, 2] * self.ti_dPdF_field[e, 8, colI]
                    self.ti_intermediate_field[e, 9, colI] = IB[2, 0] * self.ti_dPdF_field[e, 0, colI] + IB[2, 1] * self.ti_dPdF_field[
                        e, 3, colI] + IB[2, 2] * self.ti_dPdF_field[e, 6, colI]
                    self.ti_intermediate_field[e, 10, colI] = IB[2, 0] * self.ti_dPdF_field[e, 1, colI] + IB[2, 1] * \
                                                         self.ti_dPdF_field[e, 4, colI] + IB[2, 2] * self.ti_dPdF_field[
                                                             e, 7, colI]
                    self.ti_intermediate_field[e, 11, colI] = IB[2, 0] * self.ti_dPdF_field[e, 2, colI] + IB[2, 1] * \
                                                         self.ti_dPdF_field[e, 5, colI] + IB[2, 2] * self.ti_dPdF_field[
                                                             e, 8, colI]
                    self.ti_intermediate_field[e, 0, colI] = -self.ti_intermediate_field[e, 3, colI] - self.ti_intermediate_field[
                        e, 6, colI] - self.ti_intermediate_field[e, 9, colI]
                    self.ti_intermediate_field[e, 1, colI] = -self.ti_intermediate_field[e, 4, colI] - self.ti_intermediate_field[
                        e, 7, colI] - self.ti_intermediate_field[e, 10, colI]
                    self.ti_intermediate_field[e, 2, colI] = -self.ti_intermediate_field[e, 5, colI] - self.ti_intermediate_field[
                        e, 8, colI] - self.ti_intermediate_field[e, 11, colI]

                self.ti_indMap_field[e, 0] = self.vertices[e, 0] * 3
                self.ti_indMap_field[e, 1] = self.vertices[e, 0] * 3 + 1
                self.ti_indMap_field[e, 2] = self.vertices[e, 0] * 3 + 2
                self.ti_indMap_field[e, 3] = self.vertices[e, 1] * 3
                self.ti_indMap_field[e, 4] = self.vertices[e, 1] * 3 + 1
                self.ti_indMap_field[e, 5] = self.vertices[e, 1] * 3 + 2
                self.ti_indMap_field[e, 6] = self.vertices[e, 2] * 3
                self.ti_indMap_field[e, 7] = self.vertices[e, 2] * 3 + 1
                self.ti_indMap_field[e, 8] = self.vertices[e, 2] * 3 + 2
                self.ti_indMap_field[e, 9] = self.vertices[e, 3] * 3
                self.ti_indMap_field[e, 10] = self.vertices[e, 3] * 3 + 1
                self.ti_indMap_field[e, 11] = self.vertices[e, 3] * 3 + 2

                for rowI in range(12):
                    c = self.cnt[None] + e * 144 + rowI * 12 + 0
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 3], \
                                                                                    IB[0, 0] * self.ti_intermediate_field[e, rowI, 0] + \
                                                                                    IB[0, 1] * self.ti_intermediate_field[e, rowI, 3] + \
                                                                                    IB[0, 2] * self.ti_intermediate_field[e, rowI, 6]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 1
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 4], \
                                                                                    IB[0, 0] * self.ti_intermediate_field[e, rowI, 1] + \
                                                                                    IB[0, 1] * self.ti_intermediate_field[e, rowI, 4] + \
                                                                                    IB[0, 2] * self.ti_intermediate_field[e, rowI, 7]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 2
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 5], \
                                                                                    IB[0, 0] * self.ti_intermediate_field[e, rowI, 2] + \
                                                                                    IB[0, 1] * self.ti_intermediate_field[e, rowI, 5] + \
                                                                                    IB[0, 2] * self.ti_intermediate_field[e, rowI, 8]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 3
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 6], \
                                                                                    IB[1, 0] * self.ti_intermediate_field[e, rowI, 0] + \
                                                                                    IB[1, 1] * self.ti_intermediate_field[e, rowI, 3] + \
                                                                                    IB[1, 2] * self.ti_intermediate_field[e, rowI, 6]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 4
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 7], \
                                                                                    IB[1, 0] * self.ti_intermediate_field[e, rowI, 1] + \
                                                                                    IB[1, 1] * self.ti_intermediate_field[e, rowI, 4] + \
                                                                                    IB[1, 2] * self.ti_intermediate_field[e, rowI, 7]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 5
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 8], \
                                                                                    IB[1, 0] * self.ti_intermediate_field[e, rowI, 2] + \
                                                                                    IB[1, 1] * self.ti_intermediate_field[e, rowI, 5] + \
                                                                                    IB[1, 2] * self.ti_intermediate_field[e, rowI, 8]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 6
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 9], \
                                                                                    IB[2, 0] * self.ti_intermediate_field[e, rowI, 0] + \
                                                                                    IB[2, 1] * self.ti_intermediate_field[e, rowI, 3] + \
                                                                                    IB[2, 2] * self.ti_intermediate_field[e, rowI, 6]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 7
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 10], \
                                                                                    IB[2, 0] * self.ti_intermediate_field[e, rowI, 1] + \
                                                                                    IB[2, 1] * self.ti_intermediate_field[e, rowI, 4] + \
                                                                                    IB[2, 2] * self.ti_intermediate_field[e, rowI, 7]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 8
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 11], \
                                                                                    IB[2, 0] * self.ti_intermediate_field[e, rowI, 2] + \
                                                                                    IB[2, 1] * self.ti_intermediate_field[e, rowI, 5] + \
                                                                                    IB[2, 2] * self.ti_intermediate_field[e, rowI, 8]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 9
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 0], - \
                    data_mat_np[2, c - 9] - data_mat_np[2, c - 6] - data_mat_np[2, c - 3]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 10
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 1], - \
                    data_mat_np[2, c - 9] - data_mat_np[2, c - 6] - data_mat_np[2, c - 3]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 11
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], self.ti_indMap_field[e, 2], - \
                    data_mat_np[2, c - 9] - data_mat_np[2, c - 6] - data_mat_np[2, c - 3]
                R10 = IB[0, 0] * P[0, 0] + IB[0, 1] * P[0, 1] + IB[0, 2] * P[0, 2]
                R11 = IB[0, 0] * P[1, 0] + IB[0, 1] * P[1, 1] + IB[0, 2] * P[1, 2]
                R12 = IB[0, 0] * P[2, 0] + IB[0, 1] * P[2, 1] + IB[0, 2] * P[2, 2]
                R20 = IB[1, 0] * P[0, 0] + IB[1, 1] * P[0, 1] + IB[1, 2] * P[0, 2]
                R21 = IB[1, 0] * P[1, 0] + IB[1, 1] * P[1, 1] + IB[1, 2] * P[1, 2]
                R22 = IB[1, 0] * P[2, 0] + IB[1, 1] * P[2, 1] + IB[1, 2] * P[2, 2]
                R30 = IB[2, 0] * P[0, 0] + IB[2, 1] * P[0, 1] + IB[2, 2] * P[0, 2]
                R31 = IB[2, 0] * P[1, 0] + IB[2, 1] * P[1, 1] + IB[2, 2] * P[1, 2]
                R32 = IB[2, 0] * P[2, 0] + IB[2, 1] * P[2, 1] + IB[2, 2] * P[2, 2]
                data_rhs_np[self.vertices[e, 1] * 3 + 0] -= R10
                data_rhs_np[self.vertices[e, 1] * 3 + 1] -= R11
                data_rhs_np[self.vertices[e, 1] * 3 + 2] -= R12
                data_rhs_np[self.vertices[e, 2] * 3 + 0] -= R20
                data_rhs_np[self.vertices[e, 2] * 3 + 1] -= R21
                data_rhs_np[self.vertices[e, 2] * 3 + 2] -= R22
                data_rhs_np[self.vertices[e, 3] * 3 + 0] -= R30
                data_rhs_np[self.vertices[e, 3] * 3 + 1] -= R31
                data_rhs_np[self.vertices[e, 3] * 3 + 2] -= R32
                data_rhs_np[self.vertices[e, 0] * 3 + 0] -= -R10 - R20 - R30
                data_rhs_np[self.vertices[e, 0] * 3 + 1] -= -R11 - R21 - R31
                data_rhs_np[self.vertices[e, 0] * 3 + 2] -= -R12 - R22 - R32
        self.cnt[None] += self.n_elements * (self.dim + 1) * self.dim * (self.dim + 1) * self.dim

    @ti.kernel
    def save_xPrev(self):
        for i in range(self.n_vertices):
            self.xPrev[i] = self.x[i]

    @ti.kernel
    def apply_sol(self, alpha: real, data_sol: ti.ext_arr()):
        for i in range(self.n_vertices):
            for d in ti.static(range(self.dim)):
                self.x(d)[i] = self.xPrev(d)[i] + data_sol[i * self.dim + d] * alpha

    @ti.kernel
    def compute_x_xtilde(self):
        for i in range(self.n_vertices):
            self.x_xtilde[i] = self.x[i] - self.xTilde[i]

    # @ti.kernel
    def copy_sol(self, sol):
        for i in range(self.n_vertices):
            for d in ti.static(range(self.dim)):
                self.data_sol[i * self.dim + d] = sol[i * self.dim + d]

    @ti.kernel
    def compute_v(self):
        for i in range(self.n_vertices):
            self.v[i] = (self.x[i] - self.xn[i]) / self.dt
            self.del_p[i] = self.x[i] - self.xn[i]

    @ti.kernel
    def output_residual(self, data_sol: ti.ext_arr()) -> real:
        residual = 0.0
        for i in range(self.n_vertices):
            for d in ti.static(range(self.dim)):
                residual = ti.max(residual, ti.abs(data_sol[i * self.dim + d]))
        # print("PN Search Direction Residual : ", residual / self.dt)
        return residual

    def write_image(self, f):
        if self.dim == 2:
            for i in range(self.n_elements):
                for j in range(3):
                    a, b = self.vertices[i, j], self.vertices[i, (j + 1) % 3]
                    self.gui.line((self.x[a][0], self.x[a][1]),
                                  (self.x[b][0], self.x[b][1]),
                                  radius=1,
                                  color=0x4FB99F)
            self.gui.show(f'output/bunny{f:06d}.png')
        else:
            f = open(f'output/bunny{f:06d}.obj', 'w')
            for i in range(self.n_vertices):
                f.write('v %.6f %.6f %.6f\n' % (self.x[i, 0], self.x[i, 1], self.x[i, 2]))
            for [p0, p1, p2] in self.boundary_triangles_:
                f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
            f.close()

    # TODO: do not always use, still be 2d!
    # @ti.func
    def calcCenterOfMass(self, vind):
        sum = ti.Vector([0.0, 0.0])
        summ = 0.0
        for i in vind:
            for d in ti.static(range(self.dim)):
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

    # TODO: do not always use
    def extreme_test(self):
        for i in range(11, self.n_vertices):
            for d in ti.static(range(2)):
                pd = np.random.rand()
                self.x(d)[i] = 0.2 + (pd-0.5)

    # copy x to y
    @ti.kernel
    def copy(self, x: ti.template(), y: ti.template()):
        for i in x:
            y[i] = x[i]


    # TODO: this one do not need to work, since we use pd->pn
    def output_all_legacy(self, pd_dis, pn_dis, grad_E, frame, T):
        frame = str(frame).zfill(5)
        if T == 0:
            out_name = "Outputs/3Doutput"+str(self.exf_ind)+"_"+str(self.mag_ind)+"_"+frame+".txt"
        else:
            out_name = "Outputs_T/3Doutput" + str(self.exf_ind) + "_" + str(self.mag_ind) + "_" + frame + ".txt"
        if not os.path.exists(out_name):
            file = open(out_name, 'w+')
            file.close()
        ele_count = self.dim + self.dim + self.dim * self.dim + self.dim + self.dim  # pd pos + pn pos + local transform + residual + force
        out = np.ones([self.n_vertices, ele_count], dtype=float)
        self.mesh.enable_connectivity()
        for i in range(self.n_vertices):
            out[i, 0] = pd_dis[i, 0]  # pd pos
            out[i, 1] = pd_dis[i, 1]
            out[i, 2] = pd_dis[i, 2]

            out[i, 3] = pn_dis[i, 0]  # pn pos
            out[i, 4] = pn_dis[i, 1]
            out[i, 5] = pn_dis[i, 2]

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
            out[i, 4] = R[0, 0]
            out[i, 5] = R[0, 1]
            out[i, 6] = R[1, 0]
            out[i, 7] = R[1, 1]

            out[i, 8] = grad_E[i * 3]
            out[i, 9] = grad_E[i * 3 + 1]
            out[i, 9] = grad_E[i * 3 + 2]

            out[i, 10] = self.ex_force[0][0]
            out[i, 11] = self.ex_force[0][1]
            out[i, 10] = self.ex_force[0][2]
        np.savetxt(out_name, out)

    def data_one_frame(self, input_p, input_v):
        if self.exf_name is "directional":       # Directional
            self.set_dir_force_3D()
        elif self.exf_name is "ring":            # Ring
            self.set_ring_force_3D()
        elif self.exf_name is "ring_circle":
            self.set_ring_circle_force_3D()

        self.copy(input_p, self.x)
        self.copy(input_v, self.v)
        self.compute_xn_and_xTilde()
        # print("m: \n", self.m.to_numpy())
        # print("1 x: \n", self.x.to_numpy())
        # print("1 v: \n", self.v.to_numpy())
        # print("1 rest T: \n", self.restT.to_numpy())
        # print("xTilde: \n", self.xTilde.to_numpy())
        while True:
            self.data_mat.fill(0)
            self.data_rhs.fill(0)
            self.data_sol.fill(0)
            self.compute_hessian_and_gradient(self.data_mat, self.data_rhs)
            # print("cnt: ", self.cnt[None])
            # print("dara mat: \n", self.data_mat.to_numpy())
            # print("dara rhs: \n", self.data_rhs.to_numpy())
            self.data_sol = solve_linear_system3(self.data_mat, self.data_rhs,
                                                 self.n_vertices * self.dim, self.dirichlet,
                                                 self.zero.to_numpy(), False, 0, self.cnt[None])
            if self.output_residual(self.data_sol) < 1e-4 * self.dt:
                break
            E0 = self.compute_energy()
            self.save_xPrev()
            alpha = 1.0
            self.apply_sol(alpha, self.data_sol)
            E = self.compute_energy()
            while E > E0:
                alpha *= 0.5
                self.apply_sol(alpha, self.data_sol)
                E = self.compute_energy()
        self.compute_v()
        # print("del_p: \n", self.del_p.to_numpy())
        # print("v: \n", self.v.to_numpy())
        # print("x: \n", self.x.to_numpy())
        return self.del_p, self.x, self.v

    # TODO: Not always use, could be wrong
    def Run(self, is_test, frame_count):
        self.compute_restT_and_m()
        total_time = 0.0
        self.set_force()
        video_manager = ti.VideoManager(output_dir=os.getcwd() + '/results/', framerate=24, automatic_build=False)
        frame_counter = 0

        if self.dim == 2:
            gui = ti.GUI('PN Standalone', background_color=0xf7f7f7)
            filename = f'./results/frame_rest.png'
            draw_image(gui, filename, self.x.to_numpy(), self.mesh_offset, self.mesh_scale, self.vertices_, self.n_elements)
        else:
            # filename = f'./results/frame_rest.png'
            gui = ti.GUI('Model Visualizer', self.camera.res)
            gui.get_event(None)
            self.model.mesh.pos.from_numpy(self.case_info['mesh'].vertices.astype(np.float32))
            update_mesh(self.model.mesh)
            self.camera.from_mouse(gui)
            self.scene.render()
            video_manager.write_frame(self.camera.img)
            gui.set_image(self.camera.img)
            gui.show()

        for f in range(frame_count):
            total_time -= time.time()
            print("==================== Frame: ", f, " ====================")
            self.compute_xn_and_xTilde()
            while True:
                self.data_mat.fill(0)
                self.data_rhs.fill(0)
                self.data_sol.fill(0)
                self.compute_hessian_and_gradient()
                self.data_sol = solve_linear_system3(self.data_mat, self.data_rhs,
                                                           self.n_vertices * self.dim, self.dirichlet,
                                                           self.zero.to_numpy(), False, 0, self.cnt[None])
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
            # update
            self.compute_v()
            # self.compute_x_xtilde()
            total_time += time.time()
            frame_counter += 1
            filename = f'./results/frame_{frame_counter:05d}.png'
            if self.dim == 2:
                draw_image(gui, filename, self.x.to_numpy(), self.mesh_offset, self.mesh_scale, self.vertices.to_numpy(), self.n_elements)
            else:
                gui.get_event(None)
                self.model.mesh.pos.from_numpy(self.x.to_numpy())
                update_mesh(self.model.mesh)
                self.camera.from_mouse(gui)
                self.scene.render()
                video_manager.write_frame(self.camera.img)
                gui.set_image(self.camera.img)
                gui.show()

    def Run2(self):
        if self.dim == 2:
            self.x.from_numpy(self.mesh.vertices.astype(np.float64))
        if self.dim == 3:
            self.x.from_numpy(self.mesh.vertices.astype(np.float64))
        self.v.fill(0)
        if self.dim == 2:
            self.vertices.from_numpy(self.mesh.faces)
        if self.dim == 3:
            self.vertices.from_numpy(self.mesh.elements)
        self.compute_restT_and_m()
        self.zero.fill(0)

        video_manager = ti.VideoManager(output_dir=os.getcwd() + '/results/', framerate=24, automatic_build=False)
        frame_counter = 0

        # if self.dim == 2:
        #     gui = ti.GUI('PN Standalone', background_color=0xf7f7f7)
        #     filename = f'./results/frame_rest.png'
        #     draw_image(gui, filename, self.x.to_numpy(), self.mesh_offset, self.mesh_scale,
        #                self.vertices.to_numpy(), self.n_elements)
        # else:
        #     # filename = f'./results/frame_rest.png'
        #     gui = ti.GUI('Model Visualizer', self.camera.res)
        #     gui.get_event(None)
        #     self.model.mesh.pos.from_numpy(self.case_info['mesh'].vertices.astype(np.float64))
        #     update_mesh(self.model.mesh)
        #     self.camera.from_mouse(gui)
        #     self.scene.render()
        #     video_manager.write_frame(self.camera.img)
        #     gui.set_image(self.camera.img)
        #     gui.show()

        for f in range(50):
            print("==================== Frame: ", f, " ====================")
            self.compute_xn_and_xTilde()
            # print("m: \n", self.m.to_numpy())
            # print("1 x: \n", self.x.to_numpy())
            # print("1 v: \n", self.v.to_numpy())
            # print("1 rest T: \n", self.restT.to_numpy())
            # print("xTilde: \n", self.xTilde.to_numpy())
            while True:
                self.data_mat.fill(0)
                self.data_rhs.fill(0)
                self.data_sol.fill(0)
                self.ti_intermediate_field.fill(0)
                self.ti_M_field.fill(0)
                self.compute_hessian_and_gradient()
                # print("cnt: ", self.cnt[None])
                # print("dara mat: \n", self.data_mat.to_numpy())
                # print("dara rhs: \n", self.data_rhs.to_numpy())
                # if self.dim == 2:
                #     self.data_sol.from_numpy(solve_linear_system(self.data_mat.to_numpy(), self.data_rhs.to_numpy(),
                #                                                  self.n_vertices * self.dim, np.array(self.dirichlet),
                #                                                  self.zero.to_numpy(), False, 0, self.cnt[None]))
                # else:
                self.data_sol.from_numpy(solve_linear_system3(self.data_mat.to_numpy(), self.data_rhs.to_numpy(),
                                                              self.n_vertices * self.dim, np.array(self.dirichlet),
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
            self.compute_v()
            # print("del_p: \n", self.del_p.to_numpy())
            print("v: \n", self.v.to_numpy())
            print("x: \n", self.x.to_numpy())
            particle_pos = self.x.to_numpy()
            vertices_ = self.vertices.to_numpy()
            # write_image(f)
            frame_counter += 1
            filename = f'./results/frame_{frame_counter:05d}.png'
            # if self.dim == 2:
            #     draw_image(gui, filename, self.x.to_numpy(), self.mesh_offset, self.mesh_scale,
            #                self.vertices.to_numpy(), self.n_elements)
            # else:
            #     gui.get_event(None)
            #     self.model.mesh.pos.from_numpy(self.x.to_numpy())
            #     update_mesh(self.model.mesh)
            #     self.camera.from_mouse(gui)
            #     self.scene.render()
            #     video_manager.write_frame(self.camera.img)
            #     gui.set_image(self.camera.img)
            #     gui.show()