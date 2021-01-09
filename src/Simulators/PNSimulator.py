import sys, os, time
from .SimulatorBase import SimulatorBase
import taichi as ti
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.JGSL_WATER import *
from Utils.neo_hookean import fixed_corotated_first_piola_kirchoff_stress
from Utils.neo_hookean import fixed_corotated_energy
from Utils.neo_hookean import fixed_corotated_first_piola_kirchoff_stress_derivative
from Utils.math_tools import svd

class PNSimulation(SimulatorBase):
    def __init__(self, sim_info):
        super().__init__(sim_info)

        # Simulator Fields
        self.ti_pos_prev = ti.Vector.field(self.dim, self.real, self.n_vertices)
        self.ti_pos_tilde = ti.Vector.field(self.dim, self.real, self.n_vertices)
        self.ti_pos_n = ti.Vector.field(self.dim, self.real, self.n_vertices)
        self.ti_grad_pos = ti.Vector.field(self.dim, self.real, self.n_vertices)
        self.data_rhs = np.zeros(shape=(200000,), dtype=np.float64)
        self.data_mat = np.zeros(shape=(3, 20000000), dtype=np.float64)
        self.data_sol = np.zeros(shape=(200000,), dtype=np.float64)
        self.cnt = ti.field(ti.i32, shape=())
        self.ti_restT = ti.Matrix.field(self.dim, self.dim, self.real, self.n_elements)

        # Compile time optimization data structure
        self.ti_dPdF_field = ti.field(self.real, (self.n_elements, self.dim * self.dim, self.dim * self.dim))
        self.ti_intermediate_field = ti.field(self.real, (self.n_elements, self.dim * (self.dim + 1), self.dim * self.dim))
        self.ti_M_field = ti.field(self.real, (self.n_elements, self.dim * self.dim, self.dim * self.dim))
        self.ti_U_field = ti.field(self.real, (self.n_elements, self.dim * self.dim, self.dim * self.dim))
        self.ti_V_field = ti.field(self.real, (self.n_elements, self.dim * self.dim, self.dim * self.dim))
        self.ti_indMap_field = ti.field(self.real, (self.n_elements, self.dim * (self.dim + 1)))

    def initial(self):
        self.base_initial()
        self.ti_pos_prev.fill(0)
        self.ti_pos_tilde.fill(0)
        self.ti_pos_n.fill(0)
        self.ti_grad_pos.fill(0)

    @ti.kernel
    def compute_xn_and_xTilde(self):
        if ti.static(self.dim == 2):
            for i in range(self.n_vertices):
                self.ti_pos_n[i] = self.ti_pos[i]
                self.ti_pos_tilde[i] = self.ti_pos[i] + self.dt * self.ti_vel[i]
                self.ti_pos_tilde(0)[i] += self.dt * self.dt * (self.ti_ex_force[i][0] / self.ti_mass[i])
                self.ti_pos_tilde(1)[i] += self.dt * self.dt * (self.ti_ex_force[i][1] / self.ti_mass[i])
        if ti.static(self.dim == 3):
            for i in range(self.n_vertices):
                self.ti_pos_n[i] = self.ti_pos[i]
                self.ti_pos_tilde[i] = self.ti_pos[i] + self.dt * self.ti_vel[i]
                self.ti_pos_tilde(0)[i] += self.dt * self.dt * (self.ti_ex_force[i][0] / self.ti_mass[i])
                self.ti_pos_tilde(1)[i] += self.dt * self.dt * (self.ti_ex_force[i][1] / self.ti_mass[i])
                self.ti_pos_tilde(2)[i] += self.dt * self.dt * (self.ti_ex_force[i][2] / self.ti_mass[i])

    @ti.func
    def compute_T(self, i):
        if ti.static(self.dim == 2):
            ab = self.ti_pos[self.ti_elements[i][1]] - self.ti_pos[self.ti_elements[i][0]]
            ac = self.ti_pos[self.ti_elements[i][2]] - self.ti_pos[self.ti_elements[i][0]]
            return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])
        else:
            ab = self.ti_pos[self.ti_elements[i][1]] - self.ti_pos[self.ti_elements[i][0]]
            ac = self.ti_pos[self.ti_elements[i][2]] - self.ti_pos[self.ti_elements[i][0]]
            ad = self.ti_pos[self.ti_elements[i][3]] - self.ti_pos[self.ti_elements[i][0]]
            return ti.Matrix([[ab[0], ac[0], ad[0]], [ab[1], ac[1], ad[1]], [ab[2], ac[2], ad[2]]])

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
                data_mat_np[2, c] = self.ti_mass[i]
                data_rhs_np[i * self.dim + d] -= self.ti_mass[i] * (self.ti_pos(d)[i] - self.ti_pos_tilde(d)[i])
        self.cnt[None] += self.n_vertices * self.dim
        # elasticity
        for e in range(self.n_elements):
            F = self.compute_T(e) @ self.ti_restT[e].inverse()  # 2 * 2
            IB = self.ti_restT[e].inverse()  # 2 * 2
            vol0 = self.ti_restT[e].determinant() / self.dim / (self.dim - 1)
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
                    self.ti_intermediate_field[e, 0, colI] = -self.ti_intermediate_field[e, 2, colI] - \
                                                             self.ti_intermediate_field[
                                                                 e, 4, colI]
                    self.ti_intermediate_field[e, 1, colI] = -self.ti_intermediate_field[e, 3, colI] - \
                                                             self.ti_intermediate_field[
                                                                 e, 5, colI]

                self.ti_indMap_field[e, 0] = self.ti_elements[e][0] * 2
                self.ti_indMap_field[e, 1] = self.ti_elements[e][0] * 2 + 1
                self.ti_indMap_field[e, 2] = self.ti_elements[e][1] * 2
                self.ti_indMap_field[e, 3] = self.ti_elements[e][1] * 2 + 1
                self.ti_indMap_field[e, 4] = self.ti_elements[e][2] * 2
                self.ti_indMap_field[e, 5] = self.ti_elements[e][2] * 2 + 1

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
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 2], \
                                                                              self.ti_indMap_field[
                                                                                  e, colI], _000 + _101
                    c = self.cnt[None] + e * 36 + colI * 6 + 1
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 3], \
                                                                              self.ti_indMap_field[
                                                                                  e, colI], _200 + _301
                    c = self.cnt[None] + e * 36 + colI * 6 + 2
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 4], \
                                                                              self.ti_indMap_field[
                                                                                  e, colI], _010 + _111
                    c = self.cnt[None] + e * 36 + colI * 6 + 3
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 5], \
                                                                              self.ti_indMap_field[
                                                                                  e, colI], _210 + _311
                    c = self.cnt[None] + e * 36 + colI * 6 + 4
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 0], \
                                                                              self.ti_indMap_field[
                                                                                  e, colI], - _000 - _101 - _010 - _111
                    c = self.cnt[None] + e * 36 + colI * 6 + 5
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, 1], \
                                                                              self.ti_indMap_field[
                                                                                  e, colI], - _200 - _301 - _210 - _311
                data_rhs_np[self.ti_elements[e][1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
                data_rhs_np[self.ti_elements[e][1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
                data_rhs_np[self.ti_elements[e][2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
                data_rhs_np[self.ti_elements[e][2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
                data_rhs_np[self.ti_elements[e][0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[
                    1, 0] - P[
                                                                0, 1] * IB[1, 1]
                data_rhs_np[self.ti_elements[e][0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[
                    1, 0] - P[
                                                                1, 1] * IB[1, 1]
            else:
                for colI in range(9):
                    self.ti_intermediate_field[e, 3, colI] = IB[0, 0] * self.ti_dPdF_field[e, 0, colI] + IB[0, 1] * \
                                                             self.ti_dPdF_field[
                                                                 e, 3, colI] + IB[0, 2] * self.ti_dPdF_field[e, 6, colI]
                    self.ti_intermediate_field[e, 4, colI] = IB[0, 0] * self.ti_dPdF_field[e, 1, colI] + IB[0, 1] * \
                                                             self.ti_dPdF_field[
                                                                 e, 4, colI] + IB[0, 2] * self.ti_dPdF_field[e, 7, colI]
                    self.ti_intermediate_field[e, 5, colI] = IB[0, 0] * self.ti_dPdF_field[e, 2, colI] + IB[0, 1] * \
                                                             self.ti_dPdF_field[
                                                                 e, 5, colI] + IB[0, 2] * self.ti_dPdF_field[e, 8, colI]
                    self.ti_intermediate_field[e, 6, colI] = IB[1, 0] * self.ti_dPdF_field[e, 0, colI] + IB[1, 1] * \
                                                             self.ti_dPdF_field[
                                                                 e, 3, colI] + IB[1, 2] * self.ti_dPdF_field[e, 6, colI]
                    self.ti_intermediate_field[e, 7, colI] = IB[1, 0] * self.ti_dPdF_field[e, 1, colI] + IB[1, 1] * \
                                                             self.ti_dPdF_field[
                                                                 e, 4, colI] + IB[1, 2] * self.ti_dPdF_field[e, 7, colI]
                    self.ti_intermediate_field[e, 8, colI] = IB[1, 0] * self.ti_dPdF_field[e, 2, colI] + IB[1, 1] * \
                                                             self.ti_dPdF_field[
                                                                 e, 5, colI] + IB[1, 2] * self.ti_dPdF_field[e, 8, colI]
                    self.ti_intermediate_field[e, 9, colI] = IB[2, 0] * self.ti_dPdF_field[e, 0, colI] + IB[2, 1] * \
                                                             self.ti_dPdF_field[
                                                                 e, 3, colI] + IB[2, 2] * self.ti_dPdF_field[e, 6, colI]
                    self.ti_intermediate_field[e, 10, colI] = IB[2, 0] * self.ti_dPdF_field[e, 1, colI] + IB[2, 1] * \
                                                              self.ti_dPdF_field[e, 4, colI] + IB[2, 2] * \
                                                              self.ti_dPdF_field[
                                                                  e, 7, colI]
                    self.ti_intermediate_field[e, 11, colI] = IB[2, 0] * self.ti_dPdF_field[e, 2, colI] + IB[2, 1] * \
                                                              self.ti_dPdF_field[e, 5, colI] + IB[2, 2] * \
                                                              self.ti_dPdF_field[
                                                                  e, 8, colI]
                    self.ti_intermediate_field[e, 0, colI] = -self.ti_intermediate_field[e, 3, colI] - \
                                                             self.ti_intermediate_field[
                                                                 e, 6, colI] - self.ti_intermediate_field[e, 9, colI]
                    self.ti_intermediate_field[e, 1, colI] = -self.ti_intermediate_field[e, 4, colI] - \
                                                             self.ti_intermediate_field[
                                                                 e, 7, colI] - self.ti_intermediate_field[e, 10, colI]
                    self.ti_intermediate_field[e, 2, colI] = -self.ti_intermediate_field[e, 5, colI] - \
                                                             self.ti_intermediate_field[
                                                                 e, 8, colI] - self.ti_intermediate_field[e, 11, colI]

                self.ti_indMap_field[e, 0] = self.ti_elements[e][0] * 3
                self.ti_indMap_field[e, 1] = self.ti_elements[e][0] * 3 + 1
                self.ti_indMap_field[e, 2] = self.ti_elements[e][0] * 3 + 2
                self.ti_indMap_field[e, 3] = self.ti_elements[e][1] * 3
                self.ti_indMap_field[e, 4] = self.ti_elements[e][1] * 3 + 1
                self.ti_indMap_field[e, 5] = self.ti_elements[e][1] * 3 + 2
                self.ti_indMap_field[e, 6] = self.ti_elements[e][2] * 3
                self.ti_indMap_field[e, 7] = self.ti_elements[e][2] * 3 + 1
                self.ti_indMap_field[e, 8] = self.ti_elements[e][2] * 3 + 2
                self.ti_indMap_field[e, 9] = self.ti_elements[e][3] * 3
                self.ti_indMap_field[e, 10] = self.ti_elements[e][3] * 3 + 1
                self.ti_indMap_field[e, 11] = self.ti_elements[e][3] * 3 + 2

                for rowI in range(12):
                    c = self.cnt[None] + e * 144 + rowI * 12 + 0
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 3], \
                                                                              IB[0, 0] * self.ti_intermediate_field[
                                                                                  e, rowI, 0] + \
                                                                              IB[0, 1] * self.ti_intermediate_field[
                                                                                  e, rowI, 3] + \
                                                                              IB[0, 2] * self.ti_intermediate_field[
                                                                                  e, rowI, 6]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 1
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 4], \
                                                                              IB[0, 0] * self.ti_intermediate_field[
                                                                                  e, rowI, 1] + \
                                                                              IB[0, 1] * self.ti_intermediate_field[
                                                                                  e, rowI, 4] + \
                                                                              IB[0, 2] * self.ti_intermediate_field[
                                                                                  e, rowI, 7]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 2
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 5], \
                                                                              IB[0, 0] * self.ti_intermediate_field[
                                                                                  e, rowI, 2] + \
                                                                              IB[0, 1] * self.ti_intermediate_field[
                                                                                  e, rowI, 5] + \
                                                                              IB[0, 2] * self.ti_intermediate_field[
                                                                                  e, rowI, 8]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 3
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 6], \
                                                                              IB[1, 0] * self.ti_intermediate_field[
                                                                                  e, rowI, 0] + \
                                                                              IB[1, 1] * self.ti_intermediate_field[
                                                                                  e, rowI, 3] + \
                                                                              IB[1, 2] * self.ti_intermediate_field[
                                                                                  e, rowI, 6]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 4
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 7], \
                                                                              IB[1, 0] * self.ti_intermediate_field[
                                                                                  e, rowI, 1] + \
                                                                              IB[1, 1] * self.ti_intermediate_field[
                                                                                  e, rowI, 4] + \
                                                                              IB[1, 2] * self.ti_intermediate_field[
                                                                                  e, rowI, 7]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 5
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 8], \
                                                                              IB[1, 0] * self.ti_intermediate_field[
                                                                                  e, rowI, 2] + \
                                                                              IB[1, 1] * self.ti_intermediate_field[
                                                                                  e, rowI, 5] + \
                                                                              IB[1, 2] * self.ti_intermediate_field[
                                                                                  e, rowI, 8]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 6
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 9], \
                                                                              IB[2, 0] * self.ti_intermediate_field[
                                                                                  e, rowI, 0] + \
                                                                              IB[2, 1] * self.ti_intermediate_field[
                                                                                  e, rowI, 3] + \
                                                                              IB[2, 2] * self.ti_intermediate_field[
                                                                                  e, rowI, 6]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 7
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 10], \
                                                                              IB[2, 0] * self.ti_intermediate_field[
                                                                                  e, rowI, 1] + \
                                                                              IB[2, 1] * self.ti_intermediate_field[
                                                                                  e, rowI, 4] + \
                                                                              IB[2, 2] * self.ti_intermediate_field[
                                                                                  e, rowI, 7]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 8
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 11], \
                                                                              IB[2, 0] * self.ti_intermediate_field[
                                                                                  e, rowI, 2] + \
                                                                              IB[2, 1] * self.ti_intermediate_field[
                                                                                  e, rowI, 5] + \
                                                                              IB[2, 2] * self.ti_intermediate_field[
                                                                                  e, rowI, 8]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 9
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 0], - \
                                                                                  data_mat_np[2, c - 9] - data_mat_np[
                                                                                  2, c - 6] - data_mat_np[2, c - 3]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 10
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 1], - \
                                                                                  data_mat_np[2, c - 9] - data_mat_np[
                                                                                  2, c - 6] - data_mat_np[2, c - 3]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 11
                    data_mat_np[0, c], data_mat_np[1, c], data_mat_np[2, c] = self.ti_indMap_field[e, rowI], \
                                                                              self.ti_indMap_field[e, 2], - \
                                                                                  data_mat_np[2, c - 9] - data_mat_np[
                                                                                  2, c - 6] - data_mat_np[2, c - 3]
                R10 = IB[0, 0] * P[0, 0] + IB[0, 1] * P[0, 1] + IB[0, 2] * P[0, 2]
                R11 = IB[0, 0] * P[1, 0] + IB[0, 1] * P[1, 1] + IB[0, 2] * P[1, 2]
                R12 = IB[0, 0] * P[2, 0] + IB[0, 1] * P[2, 1] + IB[0, 2] * P[2, 2]
                R20 = IB[1, 0] * P[0, 0] + IB[1, 1] * P[0, 1] + IB[1, 2] * P[0, 2]
                R21 = IB[1, 0] * P[1, 0] + IB[1, 1] * P[1, 1] + IB[1, 2] * P[1, 2]
                R22 = IB[1, 0] * P[2, 0] + IB[1, 1] * P[2, 1] + IB[1, 2] * P[2, 2]
                R30 = IB[2, 0] * P[0, 0] + IB[2, 1] * P[0, 1] + IB[2, 2] * P[0, 2]
                R31 = IB[2, 0] * P[1, 0] + IB[2, 1] * P[1, 1] + IB[2, 2] * P[1, 2]
                R32 = IB[2, 0] * P[2, 0] + IB[2, 1] * P[2, 1] + IB[2, 2] * P[2, 2]
                data_rhs_np[self.ti_elements[e][1] * 3 + 0] -= R10
                data_rhs_np[self.ti_elements[e][1] * 3 + 1] -= R11
                data_rhs_np[self.ti_elements[e][1] * 3 + 2] -= R12
                data_rhs_np[self.ti_elements[e][2] * 3 + 0] -= R20
                data_rhs_np[self.ti_elements[e][2] * 3 + 1] -= R21
                data_rhs_np[self.ti_elements[e][2] * 3 + 2] -= R22
                data_rhs_np[self.ti_elements[e][3] * 3 + 0] -= R30
                data_rhs_np[self.ti_elements[e][3] * 3 + 1] -= R31
                data_rhs_np[self.ti_elements[e][3] * 3 + 2] -= R32
                data_rhs_np[self.ti_elements[e][0] * 3 + 0] -= -R10 - R20 - R30
                data_rhs_np[self.ti_elements[e][0] * 3 + 1] -= -R11 - R21 - R31
                data_rhs_np[self.ti_elements[e][0] * 3 + 2] -= -R12 - R22 - R32
        self.cnt[None] += self.n_elements * (self.dim + 1) * self.dim * (self.dim + 1) * self.dim

    def data_one_frame(self, input_p, input_v):
        self.update_force_field()
        self.copy(input_p, self.ti_pos)
        self.copy(input_v, self.ti_vel)
        self.compute_xn_and_xTilde()
        while True:
            self.data_mat.fill(0)
            self.data_rhs.fill(0)
            self.data_sol.fill(0)
            self.compute_hessian_and_gradient(self.data_mat, self.data_rhs)
