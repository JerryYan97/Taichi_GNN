import numpy as np
import taichi as ti
import sys, os, time
from scipy import sparse
from scipy.sparse.linalg import factorized
from numpy.linalg import inv
from scipy.linalg import sqrtm
from .SimulatorBase import SimulatorBase
import multiprocessing as mp
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.math_tools import svd
from Utils.utils_visualization import draw_image, update_boundary_mesh, output_3d_seq


def calcCenterOfMass(vind, dim, mass, pos):
    sum = np.zeros(dim)
    summ = 0.0
    for i in vind:
        sum += mass[i] * pos[i]
        summ += mass[i]
    sum /= summ
    return sum

def calcA_qq(q_i, dim):
    sum = np.zeros((dim, dim))
    for i in range(q_i.shape[0]):
        sum += np.outer(q_i[i], np.transpose(q_i[i]))
    return np.linalg.inv(sum)


def calcA_pq(p_i, q_i, dim):
    sum = np.zeros((dim, dim))
    for i in range(p_i.shape[0]):
        sum += np.outer(p_i[i], np.transpose(q_i[i]))
    return sum


def calcR(A_pq):
    S = sqrtm(np.dot(np.transpose(A_pq), A_pq))
    R = np.dot(A_pq, inv(S))
    return R


def get_local_trans_parallel_call(workloads_list, proc_idx, adj_list, pos, initial_rel_pos, mass, dim):
    a_final_list = []
    for vert_idx in range(workloads_list[proc_idx][0], workloads_list[proc_idx][1] + 1):
        adj_v = adj_list[vert_idx]
        init_rel_adj_pos = initial_rel_pos[adj_v, :]
        cur_adj_pos = pos[adj_v, :]
        com = calcCenterOfMass(adj_v, dim, mass, pos)
        cur_rel_pos = cur_adj_pos - com[None, :]
        A_pq = calcA_pq(cur_rel_pos, init_rel_adj_pos, dim)
        A_qq = calcA_qq(init_rel_adj_pos, dim)
        A_final = np.matmul(A_pq, A_qq).reshape((1, -1))
        a_final_list.append(A_final)
    return a_final_list


def get_local_transformation(n_vertices, mesh, pos, init_pos, mass, dim):
    pool = mp.Pool()
    # Divide workloads
    cpu_cnt = mp.cpu_count()
    works_per_proc_cnt = n_vertices // cpu_cnt
    workloads_list = []
    proc_list = []
    for i in range(cpu_cnt):
        cur_proc_workload = [i * works_per_proc_cnt, (i + 1) * works_per_proc_cnt - 1]
        if i == cpu_cnt - 1:
            cur_proc_workload[1] = n_vertices - 1
        workloads_list.append(cur_proc_workload)

    adj_list = []
    for i in range(n_vertices):
        adj_list.append(np.append(mesh.get_vertex_adjacent_vertices(i), i))

    # Parallel call
    for t in range(cpu_cnt):
        proc_list.append(pool.apply_async(func=get_local_trans_parallel_call,
                                          args=(workloads_list, t, adj_list, pos, init_pos, mass, dim,)))

    # Get results
    a_finals_list = []
    for t in range(cpu_cnt):
        a_finals_list.extend(proc_list[t].get())
    pool.close()
    pool.join()
    return a_finals_list


class PDSimulation(SimulatorBase):
    def __init__(self, sim_info):
        super().__init__(sim_info)

        # Mesh
        self.mesh.enable_connectivity()

        # Material and Parameters
        self.m_weight_positional = 1e20
        self.solver_max_iteration = 10
        self.solver_stop_residual = 0.001

        # Simulator Fields
        self.ti_volume = ti.field(self.real, self.n_elements)
        self.ti_x_new = ti.Vector.field(self.dim, self.real, self.n_vertices)
        self.ti_x_del = ti.Vector.field(self.dim, self.real, self.n_vertices)
        # self.gradE = ti.field(self.real, shape=200000)  # Keep the shape same as the rhs shape of PN
        self.ti_last_pos_new = ti.Vector.field(self.dim, self.real, self.n_vertices)
        self.ti_boundary_labels = ti.field(int, self.n_vertices)
        self.ti_Dm_inv = ti.Matrix.field(self.dim, self.dim, self.real,
                                         self.n_elements)  # The inverse of the init elements -- Dm
        self.ti_F = ti.Matrix.field(self.dim, self.dim, self.real, self.n_elements)
        self.ti_A = ti.field(self.real, (self.n_elements * 2, self.dim * self.dim, self.dim * (self.dim + 1)))
        self.ti_A_i = ti.field(self.real, shape=(self.dim * self.dim, self.dim * (self.dim + 1)))
        self.ti_q_idx_vec = ti.field(ti.int32, (self.n_elements, self.dim * (self.dim + 1)))
        self.ti_Bp = ti.Matrix.field(self.dim, self.dim, self.real, self.n_elements * 2)
        self.ti_Sn = ti.field(self.real, self.n_vertices * self.dim)
        self.ti_phi = ti.field(self.real, self.n_elements)
        self.ti_weight_strain = ti.field(self.real, self.n_elements)  # self.mu * 2 * self.volume
        self.ti_weight_volume = ti.field(self.real, self.n_elements)  # self.lam * self.dim * self.volume

        # Shape Matching
        self.ti_x_init = ti.Vector.field(self.dim, self.real, self.n_vertices)

    def initial(self):
        self.base_initial()

        # Simulator Fields
        self.ti_volume.fill(0)
        self.ti_x_new.fill(0)
        self.ti_x_del.fill(0)
        # self.gradE.fill(0)
        self.ti_last_pos_new.fill(0)
        self.ti_boundary_labels.fill(0)
        self.ti_Dm_inv.fill(0)
        self.ti_F.fill(0)
        self.ti_A.fill(0)
        self.ti_A_i.fill(0)
        self.ti_q_idx_vec.fill(0)
        self.ti_Bp.fill(0)
        self.ti_Sn.fill(0)
        self.ti_phi.fill(0)
        self.ti_weight_strain.fill(0)
        self.ti_weight_volume.fill(0)

        # Shape Matching
        self.ti_x_init.from_numpy(self.mesh.vertices)

    @ti.func
    def set_ti_A_i(self, ele_idx, row, col, val):
        self.ti_A[ele_idx, row, col] = val

    @ti.func
    def get_ti_A_i(self, ele_idx):
        if ti.static(self.dim == 2):
            tmp_mat = ti.Matrix.rows([[self.ti_A[ele_idx, 0, 0], self.ti_A[ele_idx, 0, 1], self.ti_A[ele_idx, 0, 2],
                                       self.ti_A[ele_idx, 0, 3], self.ti_A[ele_idx, 0, 4], self.ti_A[ele_idx, 0, 5]],
                                      [self.ti_A[ele_idx, 1, 0], self.ti_A[ele_idx, 1, 1], self.ti_A[ele_idx, 1, 2],
                                       self.ti_A[ele_idx, 1, 3], self.ti_A[ele_idx, 1, 4], self.ti_A[ele_idx, 1, 5]],
                                      [self.ti_A[ele_idx, 2, 0], self.ti_A[ele_idx, 2, 1], self.ti_A[ele_idx, 2, 2],
                                       self.ti_A[ele_idx, 2, 3], self.ti_A[ele_idx, 2, 4], self.ti_A[ele_idx, 2, 5]],
                                      [self.ti_A[ele_idx, 3, 0], self.ti_A[ele_idx, 3, 1], self.ti_A[ele_idx, 3, 2],
                                       self.ti_A[ele_idx, 3, 3], self.ti_A[ele_idx, 3, 4], self.ti_A[ele_idx, 3, 5]]])
            return tmp_mat
        else:
            tmp_mat = ti.Matrix.rows(
                [[self.ti_A[ele_idx, 0, 0], self.ti_A[ele_idx, 0, 1], self.ti_A[ele_idx, 0, 2],
                  self.ti_A[ele_idx, 0, 3],
                  self.ti_A[ele_idx, 0, 4], self.ti_A[ele_idx, 0, 5], self.ti_A[ele_idx, 0, 6],
                  self.ti_A[ele_idx, 0, 7],
                  self.ti_A[ele_idx, 0, 8], self.ti_A[ele_idx, 0, 9], self.ti_A[ele_idx, 0, 10],
                  self.ti_A[ele_idx, 0, 11]],
                 [self.ti_A[ele_idx, 1, 0], self.ti_A[ele_idx, 1, 1], self.ti_A[ele_idx, 1, 2],
                  self.ti_A[ele_idx, 1, 3],
                  self.ti_A[ele_idx, 1, 4], self.ti_A[ele_idx, 1, 5], self.ti_A[ele_idx, 1, 6],
                  self.ti_A[ele_idx, 1, 7],
                  self.ti_A[ele_idx, 1, 8], self.ti_A[ele_idx, 1, 9], self.ti_A[ele_idx, 1, 10],
                  self.ti_A[ele_idx, 1, 11]],
                 [self.ti_A[ele_idx, 2, 0], self.ti_A[ele_idx, 2, 1], self.ti_A[ele_idx, 2, 2],
                  self.ti_A[ele_idx, 2, 3],
                  self.ti_A[ele_idx, 2, 4], self.ti_A[ele_idx, 2, 5], self.ti_A[ele_idx, 2, 6],
                  self.ti_A[ele_idx, 2, 7],
                  self.ti_A[ele_idx, 2, 8], self.ti_A[ele_idx, 2, 9], self.ti_A[ele_idx, 2, 10],
                  self.ti_A[ele_idx, 2, 11]],
                 [self.ti_A[ele_idx, 3, 0], self.ti_A[ele_idx, 3, 1], self.ti_A[ele_idx, 3, 2],
                  self.ti_A[ele_idx, 3, 3],
                  self.ti_A[ele_idx, 3, 4], self.ti_A[ele_idx, 3, 5], self.ti_A[ele_idx, 3, 6],
                  self.ti_A[ele_idx, 3, 7],
                  self.ti_A[ele_idx, 3, 8], self.ti_A[ele_idx, 3, 9], self.ti_A[ele_idx, 3, 10],
                  self.ti_A[ele_idx, 3, 11]],
                 [self.ti_A[ele_idx, 4, 0], self.ti_A[ele_idx, 4, 1], self.ti_A[ele_idx, 4, 2],
                  self.ti_A[ele_idx, 4, 3],
                  self.ti_A[ele_idx, 4, 4], self.ti_A[ele_idx, 4, 5], self.ti_A[ele_idx, 4, 6],
                  self.ti_A[ele_idx, 4, 7],
                  self.ti_A[ele_idx, 4, 8], self.ti_A[ele_idx, 4, 9], self.ti_A[ele_idx, 4, 10],
                  self.ti_A[ele_idx, 4, 11]],
                 [self.ti_A[ele_idx, 5, 0], self.ti_A[ele_idx, 5, 1], self.ti_A[ele_idx, 5, 2],
                  self.ti_A[ele_idx, 5, 3],
                  self.ti_A[ele_idx, 5, 4], self.ti_A[ele_idx, 5, 5], self.ti_A[ele_idx, 5, 6],
                  self.ti_A[ele_idx, 5, 7],
                  self.ti_A[ele_idx, 5, 8], self.ti_A[ele_idx, 5, 9], self.ti_A[ele_idx, 5, 10],
                  self.ti_A[ele_idx, 5, 11]],
                 [self.ti_A[ele_idx, 6, 0], self.ti_A[ele_idx, 6, 1], self.ti_A[ele_idx, 6, 2],
                  self.ti_A[ele_idx, 6, 3],
                  self.ti_A[ele_idx, 6, 4], self.ti_A[ele_idx, 6, 5], self.ti_A[ele_idx, 6, 6],
                  self.ti_A[ele_idx, 6, 7],
                  self.ti_A[ele_idx, 6, 8], self.ti_A[ele_idx, 6, 9], self.ti_A[ele_idx, 6, 10],
                  self.ti_A[ele_idx, 6, 11]],
                 [self.ti_A[ele_idx, 7, 0], self.ti_A[ele_idx, 7, 1], self.ti_A[ele_idx, 7, 2],
                  self.ti_A[ele_idx, 7, 3],
                  self.ti_A[ele_idx, 7, 4], self.ti_A[ele_idx, 7, 5], self.ti_A[ele_idx, 7, 6],
                  self.ti_A[ele_idx, 7, 7],
                  self.ti_A[ele_idx, 7, 8], self.ti_A[ele_idx, 7, 9], self.ti_A[ele_idx, 7, 10],
                  self.ti_A[ele_idx, 7, 11]],
                 [self.ti_A[ele_idx, 8, 0], self.ti_A[ele_idx, 8, 1], self.ti_A[ele_idx, 8, 2],
                  self.ti_A[ele_idx, 8, 3],
                  self.ti_A[ele_idx, 8, 4], self.ti_A[ele_idx, 8, 5], self.ti_A[ele_idx, 8, 6],
                  self.ti_A[ele_idx, 8, 7],
                  self.ti_A[ele_idx, 8, 8], self.ti_A[ele_idx, 8, 9], self.ti_A[ele_idx, 8, 10],
                  self.ti_A[ele_idx, 8, 11]]
                 ])
            return tmp_mat

    @ti.func
    def compute_Dm(self, i):
        if ti.static(self.dim == 2):
            ia, ib, ic = self.ti_elements[i]
            a, b, c = self.ti_x_init[ia], self.ti_x_init[ib], self.ti_x_init[ic]
            return ti.Matrix.cols([b - a, c - a])
        else:
            idx_a, idx_b, idx_c, idx_d = self.ti_elements[i]
            a, b, c, d = self.ti_x_init[idx_a], self.ti_x_init[idx_b], self.ti_x_init[idx_c], self.ti_x_init[
                idx_d]
            return ti.Matrix.cols([b - a, c - a, d - a])

    @ti.kernel
    def init_mesh_DmInv(self, input_dirichlet: ti.ext_arr(), input_dirichlet_num: int):
        for i in range(input_dirichlet_num):
            self.ti_boundary_labels[int(input_dirichlet[i])] = 1
        for i in range(self.n_elements):
            # Compute Dm:
            Dm_i = self.compute_Dm(i)
            self.ti_Dm_inv[i] = Dm_i.inverse()
            self.ti_volume[i] = ti.abs(Dm_i.determinant()) * 0.5
            self.ti_weight_strain[i] = self.mu * 2 * self.ti_volume[i]
            self.ti_weight_volume[i] = self.lam * self.dim * self.ti_volume[i]

    @ti.func
    def fill_idx_vec(self, ele_idx):
        if ti.static(self.dim == 2):
            ia, ib, ic = self.ti_elements[ele_idx]
            ia_x_idx, ia_y_idx = ia * 2, ia * 2 + 1
            ib_x_idx, ib_y_idx = ib * 2, ib * 2 + 1
            ic_x_idx, ic_y_idx = ic * 2, ic * 2 + 1
            self.ti_q_idx_vec[ele_idx, 0], self.ti_q_idx_vec[ele_idx, 1] = ia_x_idx, ia_y_idx
            self.ti_q_idx_vec[ele_idx, 2], self.ti_q_idx_vec[ele_idx, 3] = ib_x_idx, ib_y_idx
            self.ti_q_idx_vec[ele_idx, 4], self.ti_q_idx_vec[ele_idx, 5] = ic_x_idx, ic_y_idx
        else:
            idx_a, idx_b, idx_c, idx_d = self.ti_elements[ele_idx]
            idx_a_x_idx, idx_a_y_idx, idx_a_z_idx = idx_a * 3, idx_a * 3 + 1, idx_a * 3 + 2
            idx_b_x_idx, idx_b_y_idx, idx_b_z_idx = idx_b * 3, idx_b * 3 + 1, idx_b * 3 + 2
            idx_c_x_idx, idx_c_y_idx, idx_c_z_idx = idx_c * 3, idx_c * 3 + 1, idx_c * 3 + 2
            idx_d_x_idx, idx_d_y_idx, idx_d_z_idx = idx_d * 3, idx_d * 3 + 1, idx_d * 3 + 2
            self.ti_q_idx_vec[ele_idx, 0], self.ti_q_idx_vec[ele_idx, 1], self.ti_q_idx_vec[
                ele_idx, 2] = idx_a_x_idx, idx_a_y_idx, idx_a_z_idx
            self.ti_q_idx_vec[ele_idx, 3], self.ti_q_idx_vec[ele_idx, 4], self.ti_q_idx_vec[
                ele_idx, 5] = idx_b_x_idx, idx_b_y_idx, idx_b_z_idx
            self.ti_q_idx_vec[ele_idx, 6], self.ti_q_idx_vec[ele_idx, 7], self.ti_q_idx_vec[
                ele_idx, 8] = idx_c_x_idx, idx_c_y_idx, idx_c_z_idx
            self.ti_q_idx_vec[ele_idx, 9], self.ti_q_idx_vec[ele_idx, 10], self.ti_q_idx_vec[
                ele_idx, 11] = idx_d_x_idx, idx_d_y_idx, idx_d_z_idx

    @ti.kernel
    def precomputation(self, lhs_mat_row: ti.ext_arr(), lhs_mat_col: ti.ext_arr(), lhs_mat_val: ti.ext_arr()):
        dimp = self.dim + 1
        sparse_used_idx_cnt = 0
        for e_it in range(self.n_elements):
            if ti.static(self.dim == 2):
                ia, ib, ic = self.ti_elements[e_it]
                self.ti_mass[ia] += self.ti_volume[e_it] / dimp * self.rho
                self.ti_mass[ib] += self.ti_volume[e_it] / dimp * self.rho
                self.ti_mass[ic] += self.ti_volume[e_it] / dimp * self.rho
            else:
                idx_a, idx_b, idx_c, idx_d = self.ti_elements[e_it]
                self.ti_mass[idx_a] += self.ti_volume[e_it] / dimp * self.rho
                self.ti_mass[idx_b] += self.ti_volume[e_it] / dimp * self.rho
                self.ti_mass[idx_c] += self.ti_volume[e_it] / dimp * self.rho
                self.ti_mass[idx_d] += self.ti_volume[e_it] / dimp * self.rho

        # Construct A_i matrix for every element / Build A for all the constraints:
        # Strain constraints and area constraints
        for i in range(self.n_elements):
            for t in ti.static(range(2)):
                if ti.static(self.dim == 2):
                    # Get (Dm)^-1 for this element:
                    Dm_inv_i = self.ti_Dm_inv[i]
                    a = Dm_inv_i[0, 0]
                    b = Dm_inv_i[0, 1]
                    c = Dm_inv_i[1, 0]
                    d = Dm_inv_i[1, 1]
                    # Construct A_i:
                    self.set_ti_A_i(t * self.n_elements + i, 0, 0, -a - c)
                    self.set_ti_A_i(t * self.n_elements + i, 0, 2, a)
                    self.set_ti_A_i(t * self.n_elements + i, 0, 4, c)
                    self.set_ti_A_i(t * self.n_elements + i, 1, 0, -b - d)
                    self.set_ti_A_i(t * self.n_elements + i, 1, 2, b)
                    self.set_ti_A_i(t * self.n_elements + i, 1, 4, d)
                    self.set_ti_A_i(t * self.n_elements + i, 2, 1, -a - c)
                    self.set_ti_A_i(t * self.n_elements + i, 2, 3, a)
                    self.set_ti_A_i(t * self.n_elements + i, 2, 5, c)
                    self.set_ti_A_i(t * self.n_elements + i, 3, 1, -b - d)
                    self.set_ti_A_i(t * self.n_elements + i, 3, 3, b)
                    self.set_ti_A_i(t * self.n_elements + i, 3, 5, d)

                else:
                    # Get (Dm)^-1 for this element:
                    Dm_inv_i = self.ti_Dm_inv[i]
                    e11, e12, e13 = Dm_inv_i[0, 0], Dm_inv_i[0, 1], Dm_inv_i[0, 2]
                    e21, e22, e23 = Dm_inv_i[1, 0], Dm_inv_i[1, 1], Dm_inv_i[1, 2]
                    e31, e32, e33 = Dm_inv_i[2, 0], Dm_inv_i[2, 1], Dm_inv_i[2, 2]
                    # Construct A_i:
                    self.set_ti_A_i(t * self.n_elements + i, 0, 0, -(e11 + e21 + e31))
                    self.set_ti_A_i(t * self.n_elements + i, 0, 3, e11)
                    self.set_ti_A_i(t * self.n_elements + i, 0, 6, e21)
                    self.set_ti_A_i(t * self.n_elements + i, 0, 9, e31)
                    self.set_ti_A_i(t * self.n_elements + i, 1, 0, -(e12 + e22 + e32))
                    self.set_ti_A_i(t * self.n_elements + i, 1, 3, e12)
                    self.set_ti_A_i(t * self.n_elements + i, 1, 6, e22)
                    self.set_ti_A_i(t * self.n_elements + i, 1, 9, e32)
                    self.set_ti_A_i(t * self.n_elements + i, 2, 0, -(e13 + e23 + e33))
                    self.set_ti_A_i(t * self.n_elements + i, 2, 3, e13)
                    self.set_ti_A_i(t * self.n_elements + i, 2, 6, e23)
                    self.set_ti_A_i(t * self.n_elements + i, 2, 9, e33)

                    self.set_ti_A_i(t * self.n_elements + i, 3, 1, -(e11 + e21 + e31))
                    self.set_ti_A_i(t * self.n_elements + i, 3, 4, e11)
                    self.set_ti_A_i(t * self.n_elements + i, 3, 7, e21)
                    self.set_ti_A_i(t * self.n_elements + i, 3, 10, e31)
                    self.set_ti_A_i(t * self.n_elements + i, 4, 1, -(e12 + e22 + e32))
                    self.set_ti_A_i(t * self.n_elements + i, 4, 4, e12)
                    self.set_ti_A_i(t * self.n_elements + i, 4, 7, e22)
                    self.set_ti_A_i(t * self.n_elements + i, 4, 10, e32)
                    self.set_ti_A_i(t * self.n_elements + i, 5, 1, -(e13 + e23 + e33))
                    self.set_ti_A_i(t * self.n_elements + i, 5, 4, e13)
                    self.set_ti_A_i(t * self.n_elements + i, 5, 7, e23)
                    self.set_ti_A_i(t * self.n_elements + i, 5, 10, e33)

                    self.set_ti_A_i(t * self.n_elements + i, 6, 2, -(e11 + e21 + e31))
                    self.set_ti_A_i(t * self.n_elements + i, 6, 5, e11)
                    self.set_ti_A_i(t * self.n_elements + i, 6, 8, e21)
                    self.set_ti_A_i(t * self.n_elements + i, 6, 11, e31)
                    self.set_ti_A_i(t * self.n_elements + i, 7, 2, -(e12 + e22 + e32))
                    self.set_ti_A_i(t * self.n_elements + i, 7, 5, e12)
                    self.set_ti_A_i(t * self.n_elements + i, 7, 8, e22)
                    self.set_ti_A_i(t * self.n_elements + i, 7, 11, e32)
                    self.set_ti_A_i(t * self.n_elements + i, 8, 2, -(e13 + e23 + e33))
                    self.set_ti_A_i(t * self.n_elements + i, 8, 5, e13)
                    self.set_ti_A_i(t * self.n_elements + i, 8, 8, e23)
                    self.set_ti_A_i(t * self.n_elements + i, 8, 11, e33)

        # Sparse modification Changed:
        for ele_idx in range(self.n_elements):
            self.fill_idx_vec(ele_idx)
            ele_global_start_idx = ele_idx * self.dim * (self.dim + 1) * self.dim * (self.dim + 1)
            ele_offset_idx = 0
            for A_row_idx in range(self.dim * (self.dim + 1)):
                for A_col_idx in range(self.dim * (self.dim + 1)):
                    lhs_row_idx = self.ti_q_idx_vec[ele_idx, A_row_idx]
                    lhs_col_idx = self.ti_q_idx_vec[ele_idx, A_col_idx]
                    weight_strain = self.ti_weight_strain[ele_idx]
                    weight_volume = self.ti_weight_volume[ele_idx]
                    cur_sparse_val = 0.0
                    if ti.static(self.dim) == 2:
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 0, A_row_idx] * self.ti_A[ele_idx, 0, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 1, A_row_idx] * self.ti_A[ele_idx, 1, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 2, A_row_idx] * self.ti_A[ele_idx, 2, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 3, A_row_idx] * self.ti_A[ele_idx, 3, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 0, A_row_idx] * self.ti_A[ele_idx, 0, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 1, A_row_idx] * self.ti_A[ele_idx, 1, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 2, A_row_idx] * self.ti_A[ele_idx, 2, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 3, A_row_idx] * self.ti_A[ele_idx, 3, A_col_idx] * weight_volume)
                    else:
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 0, A_row_idx] * self.ti_A[ele_idx, 0, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 1, A_row_idx] * self.ti_A[ele_idx, 1, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 2, A_row_idx] * self.ti_A[ele_idx, 2, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 3, A_row_idx] * self.ti_A[ele_idx, 3, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 4, A_row_idx] * self.ti_A[ele_idx, 4, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 5, A_row_idx] * self.ti_A[ele_idx, 5, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 6, A_row_idx] * self.ti_A[ele_idx, 6, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 7, A_row_idx] * self.ti_A[ele_idx, 7, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 8, A_row_idx] * self.ti_A[ele_idx, 8, A_col_idx] * weight_strain)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 0, A_row_idx] * self.ti_A[ele_idx, 0, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 1, A_row_idx] * self.ti_A[ele_idx, 1, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 2, A_row_idx] * self.ti_A[ele_idx, 2, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 3, A_row_idx] * self.ti_A[ele_idx, 3, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 4, A_row_idx] * self.ti_A[ele_idx, 4, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 5, A_row_idx] * self.ti_A[ele_idx, 5, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 6, A_row_idx] * self.ti_A[ele_idx, 6, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 7, A_row_idx] * self.ti_A[ele_idx, 7, A_col_idx] * weight_volume)
                        cur_sparse_val += (
                                    self.ti_A[ele_idx, 8, A_row_idx] * self.ti_A[ele_idx, 8, A_col_idx] * weight_volume)
                    # lhs_matrix_np[lhs_row_idx, lhs_col_idx] += cur_sparse_val
                    cur_idx = ele_global_start_idx + ele_offset_idx
                    lhs_mat_row[cur_idx] = lhs_row_idx
                    lhs_mat_col[cur_idx] = lhs_col_idx
                    lhs_mat_val[cur_idx] = cur_sparse_val
                    ele_offset_idx += 1
        sparse_used_idx_cnt += self.n_elements * self.dim * (self.dim + 1) * self.dim * (self.dim + 1)

        # Add positional constraints and mass terms to the lhs matrix
        for i in range(self.n_vertices):
            global_start_idx = sparse_used_idx_cnt + i * self.dim
            local_offset_idx = 0
            for d in range(self.dim):
                cur_sparse_val = 0.0
                cur_sparse_val += (self.ti_mass[i] / (self.dt * self.dt))
                if self.ti_boundary_labels[i] == 1:
                    cur_sparse_val += self.m_weight_positional
                lhs_row_idx, lhs_col_idx = i * self.dim + d, i * self.dim + d
                cur_idx = global_start_idx + local_offset_idx
                lhs_mat_row[cur_idx] = lhs_row_idx
                lhs_mat_col[cur_idx] = lhs_col_idx
                lhs_mat_val[cur_idx] = cur_sparse_val
                local_offset_idx += 1

    @ti.func
    def compute_Fi(self, i):
        if ti.static(self.dim == 2):
            ia, ib, ic = self.ti_elements[i]
            a, b, c = self.ti_x_new[ia], self.ti_x_new[ib], self.ti_x_new[ic]
            D_i = ti.Matrix.cols([b - a, c - a])
            return ti.cast(D_i @ self.ti_Dm_inv[i], self.real)
        else:
            idx_a, idx_b, idx_c, idx_d = self.ti_elements[i]
            a, b, c, d = self.ti_x_new[idx_a], self.ti_x_new[idx_b], self.ti_x_new[idx_c], self.ti_x_new[idx_d]
            D_i = ti.Matrix.cols([b - a, c - a, d - a])
            return ti.cast(D_i @ self.ti_Dm_inv[i], self.real)

    @ti.func
    def compute_volume_constraint(self, sigma):
        if ti.static(self.dim == 2):
            sigma_star_11_k, sigma_star_22_k = 1.0, 1.0
            for itr in ti.static(range(10)):
                first_term = (sigma_star_11_k * sigma_star_22_k - sigma[0, 0] * sigma_star_22_k - sigma[1, 1]
                              * sigma_star_11_k + 1.0) / (sigma_star_11_k ** 2 + sigma_star_22_k ** 2)
                D1_kplus1 = first_term * sigma_star_22_k
                D2_kplus1 = first_term * sigma_star_11_k
                sigma_star_11_k = D1_kplus1 + sigma[0, 0]
                sigma_star_22_k = D2_kplus1 + sigma[1, 1]
            return ti.Matrix.rows([[sigma_star_11_k, 0.0], [0.0, sigma_star_22_k]])
        else:
            sigma_star_11_k, sigma_star_22_k, sigma_star_33_k = 1.0, 1.0, 1.0
            for itr in ti.static(range(10)):
                D_k = ti.Vector([sigma_star_11_k - sigma[0, 0],
                                 sigma_star_22_k - sigma[1, 1],
                                 sigma_star_33_k - sigma[2, 2]])
                C_D_k = sigma_star_11_k * sigma_star_22_k * sigma_star_33_k - 1.0
                grad_C_D_k = ti.Vector([sigma_star_22_k * sigma_star_33_k,
                                        sigma_star_11_k * sigma_star_33_k,
                                        sigma_star_11_k * sigma_star_22_k])
                first_term = (grad_C_D_k.dot(D_k) - C_D_k) / (grad_C_D_k.norm() ** 2)
                D_kplus1 = first_term * D_k
                sigma_star_11_k = D_kplus1[0] + sigma[0, 0]
                sigma_star_22_k = D_kplus1[1] + sigma[1, 1]
                sigma_star_33_k = D_kplus1[2] + sigma[2, 2]
            return ti.Matrix.rows(
                [[sigma_star_11_k, 0.0, 0.0], [0.0, sigma_star_22_k, 0.0], [0.0, 0.0, sigma_star_33_k]])

    # NOTE: This function doesn't build all constraints
    # It just builds strain constraints and area/volume constraints
    @ti.kernel
    def local_solve_build_bp_for_all_constraints(self):
        for i in range(self.n_elements):
            # Construct strain constraints:
            # Construct Current F_i:
            F_i = self.compute_Fi(i)
            self.ti_F[i] = F_i
            # Use current F_i construct current 'B * p' or Ri
            U, sigma, V = svd(F_i)
            self.ti_Bp[i] = U @ V.transpose()

            # Construct volume preservation constraints:
            # My test:
            PP = self.compute_volume_constraint(sigma)
            self.ti_Bp[self.n_elements + i] = U @ PP @ V.transpose()

        # Calculate Phi for all the elements:
        for i in range(self.n_elements):
            Bp_i_strain = self.ti_Bp[i]
            Bp_i_volume = self.ti_Bp[self.n_elements + i]
            F_i = self.ti_F[i]
            energy1 = self.mu * self.ti_volume[i] * ((F_i - Bp_i_strain).norm() ** 2)
            energy2 = 0.5 * self.lam * self.ti_volume[i] * ((F_i - Bp_i_volume).trace() ** 2)
            self.ti_phi[i] = energy1 + energy2

    @ti.kernel
    def build_sn(self):
        for vert_idx in range(self.n_vertices):  # number of vertices
            Sn_idx1 = vert_idx * self.dim
            Sn_idx2 = vert_idx * self.dim + 1
            pos_i = self.ti_x[vert_idx]
            vel_i = self.ti_vel[vert_idx]
            self.ti_Sn[Sn_idx1] = pos_i[0] + self.dt * vel_i[0] + (self.dt ** 2) * self.ti_ex_force[vert_idx][0] / \
                                  self.ti_mass[vert_idx]  # x-direction;
            self.ti_Sn[Sn_idx2] = pos_i[1] + self.dt * vel_i[1] + (self.dt ** 2) * self.ti_ex_force[vert_idx][1] / \
                                  self.ti_mass[vert_idx]  # y-direction;
            if ti.static(self.dim == 3):
                Sn_idx3 = vert_idx * self.dim + 2
                self.ti_Sn[Sn_idx3] = pos_i[2] + self.dt * vel_i[2] + (self.dt ** 2) * self.ti_ex_force[vert_idx][2] / \
                                      self.ti_mass[vert_idx]

    @ti.func
    def Build_Bp_i_vec(self, idx):
        Bp_i = self.ti_Bp[idx]
        if ti.static(self.dim == 2):
            Bp_i_vec = ti.Vector([Bp_i[0, 0], Bp_i[0, 1],
                                  Bp_i[1, 0], Bp_i[1, 1]])
            return Bp_i_vec
        else:
            Bp_i_vec = ti.Vector([Bp_i[0, 0], Bp_i[0, 1], Bp_i[0, 2],
                                  Bp_i[1, 0], Bp_i[1, 1], Bp_i[1, 2],
                                  Bp_i[2, 0], Bp_i[2, 1], Bp_i[2, 2]])
            return Bp_i_vec

    @ti.kernel
    def build_rhs(self, rhs: ti.ext_arr()):
        one_over_dt2 = 1.0 / (self.dt ** 2)
        # Construct the first part of the rhs
        for i in range(self.n_vertices * self.dim):
            rhs[i] = one_over_dt2 * self.ti_mass[i // self.dim] * self.ti_Sn[i]
        # Add strain and volume/area constraints to the rhs
        for t in ti.static(range(2)):
            for ele_idx in range(self.n_elements):
                Bp_i_vec = self.Build_Bp_i_vec(t * self.n_elements + ele_idx)
                A_i = self.get_ti_A_i(ele_idx)
                AT_Bp = A_i.transpose() @ Bp_i_vec
                weight = 0.0
                if t == 0:
                    weight = self.ti_weight_strain[ele_idx]
                else:
                    weight = self.ti_weight_volume[ele_idx]
                AT_Bp *= weight  # m_weight_strain

                if ti.static(self.dim == 2):
                    ia, ib, ic = self.ti_elements[ele_idx]

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
                    idx_a, idx_b, idx_c, idx_d = self.ti_elements[ele_idx]
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
        for i in range(self.n_vertices):
            if self.ti_boundary_labels[i] == 1:
                pos_init_i = self.ti_x_init[i]
                q_i_x_idx = i * self.dim
                q_i_y_idx = i * self.dim + 1
                rhs[q_i_x_idx] += (pos_init_i[0] * self.m_weight_positional)
                rhs[q_i_y_idx] += (pos_init_i[1] * self.m_weight_positional)
                if ti.static(self.dim == 3):
                    q_i_z_idx = i * self.dim + 2
                    rhs[q_i_z_idx] += (pos_init_i[2] * self.m_weight_positional)

    @ti.kernel
    def update_velocity_pos(self):
        for i in range(self.n_vertices):
            self.ti_vel[i] = (self.ti_x_new[i] - self.ti_x[i]) / self.dt
            self.ti_x_del[i] = self.ti_x_new[i] - self.ti_x[i]
            self.ti_x[i] = self.ti_x_new[i]

    @ti.kernel
    def warm_up(self):
        for pos_idx in range(self.n_vertices):
            sn_idx1, sn_idx2 = pos_idx * self.dim, pos_idx * self.dim + 1
            self.ti_x_new[pos_idx][0] = self.ti_Sn[sn_idx1]
            self.ti_x_new[pos_idx][1] = self.ti_Sn[sn_idx2]
            if ti.static(self.dim == 3):
                sn_idx3 = pos_idx * self.dim + 2
                self.ti_x_new[pos_idx][2] = self.ti_Sn[sn_idx3]

    @ti.kernel
    def update_pos_new_from_numpy(self, sol: ti.ext_arr()):
        for pos_idx in range(self.n_vertices):
            sol_idx1, sol_idx2 = pos_idx * self.dim, pos_idx * self.dim + 1
            self.ti_x_new[pos_idx][0] = sol[sol_idx1]
            self.ti_x_new[pos_idx][1] = sol[sol_idx2]
            if ti.static(self.dim == 3):
                sol_idx3 = pos_idx * self.dim + 2
                self.ti_x_new[pos_idx][2] = sol[sol_idx3]

    @ti.kernel
    def check_residual(self) -> ti.f64:
        residual = 0.0
        for i in range(self.n_vertices):
            residual += (self.ti_last_pos_new[i] - self.ti_x_new[i]).norm()
            self.ti_last_pos_new[i] = self.ti_x_new[i]
        # print("residual:", residual)
        return residual

    def output_network_data(self, pd_dis, pn_dis, gradE, init_rel_pos, frame, T):
        vel = self.ti_vel.to_numpy()
        # gradE = grad_E.to_numpy()
        if self.dim == 2:
            exf = np.array([self.ti_ex_force[0][0], self.ti_ex_force[0][1]])
        else:
            exf = np.array([self.ti_ex_force[0][0], self.ti_ex_force[0][1], self.ti_ex_force[0][2]])

        frame = str(frame).zfill(5)
        if T == 0:
            if self.dim == 2:
                out_name = "SimData/TrainingData/Train_2d_" + self.case_info['case_name'] + "_" + str(self.exf_angle) + \
                           "_" + str(self.exf_mag) + "_" + frame + ".csv"
            else:
                if self.force_type == 'dir':
                    out_name = "SimData/TrainingData/Train_dir_" + self.case_info['case_name'] + "_" + \
                               str(self.exf_angle1) + "_" + str(self.exf_angle2) + "_" + str(self.exf_mag) + \
                               "_" + frame + ".csv"
                elif self.force_type == 'ring':
                    out_name = "SimData/TrainingData/Train_ring_" + self.case_info['case_name'] + "_" +\
                               str(self.ring_mag) + "_" + str(self.ring_width) + "_" + str(self.ring_angle) + \
                               "_" + frame + ".csv"
        else:
            if self.dim == 2:
                out_name = "SimData/TestingData/Test_2d_" + self.case_info['case_name'] + "_" + str(self.exf_angle) + \
                            "_" + str(self.exf_mag) + "_" + frame + ".csv"
            else:
                if self.force_type == 'dir':
                    out_name = "SimData/TestingData/Test_dir_" + self.case_info['case_name'] + "_" + \
                               str(self.exf_angle1) + "_" + str(self.exf_angle2) + "_" + str(self.exf_mag) + \
                               "_" + frame + ".csv"
                elif self.force_type == 'ring':
                    out_name = "SimData/TestingData/Test_ring_" + self.case_info['case_name'] + "_" + \
                               str(self.ring_mag) + "_" + str(self.ring_width) + "_" + str(self.ring_angle) + \
                               "_" + frame + ".csv"

        ele_count = self.dim + self.dim + self.dim * self.dim + self.dim + self.dim + self.dim + self.dim
        out = np.ones([self.n_vertices, ele_count], dtype=float)
        ltrans_start_t = time.time()
        A_finals = get_local_transformation(self.n_vertices, self.mesh, self.ti_x.to_numpy(), init_rel_pos,
                                            self.ti_mass.to_numpy(), self.dim)
        ltrans_end_t = time.time()
        print("get local transformation:", ltrans_end_t - ltrans_start_t)
        i = 0
        pos_init_out = self.mesh.vertices
        if self.dim == 2:
            for res in A_finals:
                out[i, 0:2] = pd_dis[i, :]  # pd pos
                out[i, 2:4] = pn_dis[i, :]  # pn pos
                out[i, 4:8] = res
                out[i, 8:10] = gradE[i * 2: i * 2 + 2]
                out[i, 10:12] = exf
                out[i, 12:14] = vel[i, :]
                out[i, 14:16] = pos_init_out[i, :]
                i = i + 1
        else:
            for res in A_finals:
                out[i, 0:3] = pd_dis[i, :]  # pd pos
                out[i, 3:6] = pn_dis[i, :]  # pn pos
                out[i, 6:15] = res
                out[i, 15:18] = gradE[i * 3: i * 3 + 3]
                out[i, 18:21] = exf
                out[i, 21:24] = vel[i, :]
                out[i, 24:27] = pos_init_out[i, :]
                i = i + 1

        fill_data_end = time.time()
        print("fill data: ", fill_data_end - ltrans_end_t)
        np.savetxt(out_name, out, delimiter=',')

    def output_aux_data(self, f, pn_pos):
        if self.dim == 3:
            if self.force_type == 'dir':
                name_pd = "SimData/PDAnimSeq/PD_dir_" + self.case_info['case_name'] + "_" + str(self.exf_angle1) + "_" + \
                          str(self.exf_angle2) + "_"+str(self.exf_mag)+"_"+str(f).zfill(6)+".obj"
                name_pn = "SimData/PNAnimSeq/PN_dir" + self.case_info['case_name'] + "_" + str(self.exf_angle1) + "_" + \
                          str(self.exf_angle2) + "_"+str(self.exf_mag)+"_"+str(f).zfill(6)+".obj"
            elif self.force_type == 'ring':
                name_pd = "SimData/PDAnimSeq/PD_ring_" + self.case_info['case_name'] + "_" + str(self.ring_mag) + "_" + \
                          str(self.ring_width) + "_" + str(self.ring_angle) + "_" + str(f).zfill(6) + ".obj"
                name_pn = "SimData/PNAnimSeq/PN_ring_" + self.case_info['case_name'] + "_" + str(self.ring_mag) + "_" + \
                          str(self.ring_width) + "_" + str(self.ring_angle) + "_" + str(f).zfill(6) + ".obj"
            elif self.force_type == 'ring_circle':
                name_pd = "SimData/PDAnimSeq/PD_ringC_" + self.case_info['case_name'] + "_" + str(self.ring_mag) + "_" + \
                          str(self.ring_width) + "_" + str(self.ring_angle) + "_" + str(f).zfill(6) + ".obj"
                name_pn = "SimData/PNAnimSeq/PN_ringC_" + self.case_info['case_name'] + "_" + str(self.ring_mag) + "_" + \
                          str(self.ring_width) + "_" + str(self.ring_angle) + "_" + str(f).zfill(6) + ".obj"

            output_3d_seq(self.ti_x.to_numpy(), self.boundary_triangles, name_pd)
            output_3d_seq(pn_pos.to_numpy(), self.boundary_triangles, name_pn)

    def run(self, pn, is_test, frame_count, scene_info):
        rhs_np = np.zeros(self.n_vertices * self.dim, dtype=np.float64)
        lhs_mat_val = np.zeros(
            shape=(self.n_elements * self.dim ** 2 * (self.dim + 1) ** 2 + self.n_vertices * self.dim,),
            dtype=np.float64)
        lhs_mat_row = np.zeros(
            shape=(self.n_elements * self.dim ** 2 * (self.dim + 1) ** 2 + self.n_vertices * self.dim,),
            dtype=np.float64)
        lhs_mat_col = np.zeros(
            shape=(self.n_elements * self.dim ** 2 * (self.dim + 1) ** 2 + self.n_vertices * self.dim,),
            dtype=np.float64)

        self.init_mesh_DmInv(self.dirichlet, len(self.dirichlet))
        self.precomputation(lhs_mat_row, lhs_mat_col, lhs_mat_val)
        s_lhs_matrix_np = sparse.csr_matrix((lhs_mat_val, (lhs_mat_row, lhs_mat_col)),
                                            shape=(self.n_vertices * self.dim, self.n_vertices * self.dim),
                                            dtype=np.float64)
        pre_fact_lhs_solve = factorized(s_lhs_matrix_np)
        if self.dim == 2:
            gui = scene_info['gui']
            filename = f'./SimData/TmpRenderedImgs/frame_rest.png'
            draw_image(gui, filename, self.ti_x.to_numpy(), self.mesh_offset, self.mesh_scale,
                       self.ti_elements.to_numpy(), self.n_elements)
        else:
            gui = scene_info['gui']
            scene_info['model'].set_transform(self.case_info['transformation_mat'])
            update_boundary_mesh(self.ti_x, scene_info['boundary_pos'], self.case_info)
            scene_info['scene'].input(gui)
            scene_info['tina_mesh'].set_face_verts(scene_info['boundary_pos'])
            scene_info['scene'].render()
            gui.set_image(scene_info['scene'].img)
            gui.show()

        frame_counter = 0
        init_com = calcCenterOfMass(np.arange(self.n_vertices),
                                    self.dim, self.ti_mass.to_numpy(),
                                    self.ti_x.to_numpy())  # this is right
        init_rel_pos = self.mesh.vertices - init_com
        while frame_counter < frame_count:
            print("//////////////////////////////////////Frame ", frame_counter, "/////////////////////////////////")
            frame_start_t = time.time()
            self.update_force_field()
            self.build_sn()
            self.warm_up()

            pn_start_t = time.time()
            pn_dis, _pn_pos, pn_v = pn.data_one_frame(self.ti_x, self.ti_vel)
            pn_end_t = time.time()
            print("pn solve time: ", pn_end_t - pn_start_t)

            pd_start_t = time.time()
            for itr in range(self.solver_max_iteration):
                self.local_solve_build_bp_for_all_constraints()
                self.build_rhs(rhs_np)

                pos_new_np = pre_fact_lhs_solve(rhs_np)
                self.update_pos_new_from_numpy(pos_new_np)

                residual = self.check_residual()
                if residual < self.solver_stop_residual:
                    break
            pd_end_t = time.time()
            print("pd solve time: ", pd_end_t - pd_start_t)

            self.update_velocity_pos()
            # self.gradE.from_numpy(pn.get_gradE_from_pd(self.ti_x))
            gradE = pn.get_gradE_from_pd(self.ti_x)
            t_out_start = time.time()
            self.output_network_data(self.ti_x_del.to_numpy(),
                                     pn_dis.to_numpy(),
                                     gradE, init_rel_pos,
                                     frame_counter, is_test)
            t_out_end = time.time()
            print("output network data time: ", t_out_end - t_out_start)

            frame_counter += 1
            frame_end_t = time.time()
            print("whole time for one frame: ", frame_end_t - frame_start_t)

            # Show result
            if self.dim == 2:
                filename = f'./SimData/TmpRenderedImgs/frame_{frame_counter:05d}.png'
                draw_image(gui, filename, self.ti_x.to_numpy(), self.mesh_offset, self.mesh_scale,
                           self.ti_elements.to_numpy(), self.n_elements)
            else:
                # update_boundary_mesh(self.ti_x, scene_info['boundary_pos'], self.case_info)
                update_boundary_mesh(_pn_pos, scene_info['boundary_pos'], self.case_info)
                scene_info['scene'].input(gui)
                scene_info['tina_mesh'].set_face_verts(scene_info['boundary_pos'])
                scene_info['scene'].render()
                gui.set_image(scene_info['scene'].img)
                gui.show()

            self.output_aux_data(frame_counter, _pn_pos)
