import sys, os, time
import taichi as ti
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.utils_visualization import draw_image, update_boundary_mesh, get_force_field, output_3d_seq
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy import sparse
from scipy.sparse.linalg import factorized
from Utils.math_tools import svd
import multiprocessing as mp

real = ti.f64


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


def f(adj_v, pos, initial_rel_pos, mass, dim):
    init_rel_pos = initial_rel_pos[adj_v, :]
    new_pos = pos[adj_v, :]
    com = calcCenterOfMass(adj_v, dim, mass, pos)
    curr_rel_pos = new_pos - com[None, :]
    A_pq = calcA_pq(curr_rel_pos, init_rel_pos, dim)
    A_qq = calcA_qq(init_rel_pos, dim)
    A_final = np.matmul(A_pq, A_qq).reshape((1, -1))
    return A_final


def get_local_transformation(n_vertices, mesh, pos, init_pos, mass, dim):
    cores = mp.cpu_count()
    Pool = mp.Pool(processes=cores)
    adj_list = []
    for i in range(n_vertices):
        adj_list.append(np.sort(np.append(mesh.get_vertex_adjacent_vertices(i), i)))
    multi_res = [Pool.apply_async(f, (adj, pos, init_pos, mass, dim)) for adj in adj_list]
    return multi_res


@ti.data_oriented
class PDSimulation:
    def __init__(self, case_info, _dt):
        self.dt = _dt
        ################################ mesh ######################################
        self.case_info = case_info
        self.mesh = self.case_info['mesh']
        self.dirichlet = self.case_info['dirichlet']
        self.mesh_scale = self.case_info['mesh_scale']
        self.mesh_offset = self.case_info['mesh_offset']
        self.dim = self.case_info['dim']
        self.n_vertices = self.mesh.num_vertices
        if self.dim == 2:
            self.n_elements = self.mesh.num_faces
        else:
            self.n_elements = self.mesh.num_elements

        ################################ material and parms ######################################
        self.rho = 100
        self.E = 1e4
        self.nu = 0.4
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.m_weight_positional = 1e20
        self.solver_max_iteration = 50
        self.solver_stop_residual = 0.0001
        ################################ field ######################################
        self.ti_pos_del = ti.Vector.field(self.dim, real, self.n_vertices)
        self.gradE = ti.field(real, shape=2000)
        self.x_xtilde = ti.Vector.field(self.dim, real, self.n_vertices)
        ################################ shape matching ######################################
        self.initial_rel_pos = np.array([self.n_vertices, self.dim])
        self.pi = ti.Vector.field(self.dim, real, self.n_vertices)
        self.qi = ti.Vector.field(self.dim, real, self.n_vertices)
        self.init_pos = self.mesh.vertices.astype(np.float32)
        self.init_pos = self.init_pos[:, :self.dim]
        ################################ force field ################################
        self.ti_ex_force = ti.Vector.field(self.dim, real, 1)
        self.npex_f = np.zeros((self.dim, 1))
        self.exf_angle1 = 0.0
        self.exf_angle2 = 0.0
        self.exf_mag = 0.0

        self.input_xn = ti.Vector.field(self.dim, real, self.n_vertices)
        self.input_vn = ti.Vector.field(self.dim, real, self.n_vertices)

        # Taichi variables' initialization:
        self.ti_mass = ti.field(real, self.n_vertices)
        self.ti_volume = ti.field(real, self.n_elements)
        self.ti_pos = ti.Vector.field(self.dim, real, self.n_vertices)
        self.ti_pos_new = ti.Vector.field(self.dim, real, self.n_vertices)
        self.ti_pos_init = ti.Vector.field(self.dim, real, self.n_vertices)
        self.ti_last_pos_new = ti.Vector.field(self.dim, real, self.n_vertices)
        self.ti_boundary_labels = ti.field(int, self.n_vertices)
        self.ti_vel = ti.Vector.field(self.dim, real, self.n_vertices)
        self.ti_vel_last = ti.Vector.field(self.dim, real, self.n_vertices)

        self.ti_elements = ti.Vector.field(self.dim + 1, int, self.n_elements)  # ids of three vertices of each face
        self.ti_Dm_inv = ti.Matrix.field(self.dim, self.dim, real, self.n_elements)  # The inverse of the init elements -- Dm
        self.ti_F = ti.Matrix.field(self.dim, self.dim, real, self.n_elements)
        # ti_A = ti.Matrix.field(dim * dim, dim * (dim + 1), real, n_elements * 2)
        self.ti_A = ti.field(real, (self.n_elements * 2, self.dim * self.dim, self.dim * (self.dim + 1)))
        self.ti_A_i = ti.field(real, shape=(self.dim * self.dim, self.dim * (self.dim + 1)))
        # A_i = ti.field(real, shape=(dim * dim, dim * (dim + 1)))
        self.ti_q_idx_vec = ti.field(ti.int32, (self.n_elements, self.dim * (self.dim + 1)))
        self.ti_Bp = ti.Matrix.field(self.dim, self.dim, real, self.n_elements * 2)
        # ti_rhs_np = np.zeros(n_vertices * dim, dtype=np.float64)
        self.ti_Sn = ti.field(real, self.n_vertices * self.dim)
        # self.ti_lhs_matrix = ti.field(real, shape=(self.n_vertices * self.dim, self.n_vertices * self.dim))
        # potential energy of each element(face) for linear coratated elasticity material.
        self.ti_phi = ti.field(real, self.n_elements)
        self.ti_weight_strain = ti.field(real, self.n_elements)   # self.mu * 2 * self.volume
        self.ti_weight_volume = ti.field(real, self.n_elements)   # self.lam * self.dim * self.volume
        self.boundary_points, self.boundary_edges, self.boundary_triangles = self.case_info['boundary']
        self.mesh.enable_connectivity()

        self.pos_init_out = []

    def initial(self):
        if self.dim == 2:
            self.initial_com = ti.Vector([0.0, 0.0])
        else:
            self.initial_com = ti.Vector([0.0, 0.0, 0.0])

        if self.dim == 2:
            self.ti_elements.from_numpy(self.mesh.faces)
        else:
            self.ti_elements.from_numpy(self.mesh.elements)

        self.ti_pos.from_numpy(self.mesh.vertices)
        self.ti_pos_init.from_numpy(self.mesh.vertices)
        self.pos_init_out = self.ti_pos_init.to_numpy()
        self.input_xn.fill(0)
        self.input_vn.fill(0)
        self.ti_mass.fill(0)
        self.ti_volume.fill(0)
        self.ti_pos_new.fill(0)
        self.ti_last_pos_new.fill(0)
        self.ti_boundary_labels.fill(0)
        self.ti_vel.fill(0)
        self.ti_vel_last.fill(0)
        self.ti_Dm_inv.fill(0)
        self.ti_F.fill(0)
        self.ti_A.fill(0)
        self.ti_A_i.fill(0)
        self.ti_q_idx_vec.fill(0)
        self.ti_Bp.fill(0)
        self.ti_Sn.fill(0)
        # self.ti_lhs_matrix.fill(0)
        self.ti_phi.fill(0)
        self.ti_weight_strain.fill(0)
        self.ti_weight_volume.fill(0)
        self.ti_pos_del.fill(0)
        self.gradE.fill(0)
        self.x_xtilde.fill(0)
        ################################ shape matching ######################################
        self.pi.fill(0)
        self.qi.fill(0)
        ################################ force field ################################
        self.ti_ex_force.fill(0)


    def set_material(self, _rho, _ym, _nu, _dt):
        self.dt = _dt
        self.rho = _rho
        self.E = _ym
        self.nu = _nu
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    def set_force(self, ang1, ang2, mag):
        if self.dim == 2:
            exf_angle = -45.0
            exf_mag = 6
            self.ti_ex_force[0] = ti.Vector(get_force_field(exf_mag, exf_angle))
        else:
            # exf_angle1 = 45.0
            # exf_angle2 = 45.0
            # exf_mag = 6
            self.exf_angle1 = ang1
            self.exf_angle2 = ang2
            self.exf_mag = mag
            self.ti_ex_force[0] = ti.Vector(get_force_field(self.exf_mag, self.exf_angle1, self.exf_angle2, 3))
            print("force: ", self.ti_ex_force[0][0], " ", self.ti_ex_force[0][1], " ", self.ti_ex_force[0][2])

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
                [[self.ti_A[ele_idx, 0, 0], self.ti_A[ele_idx, 0, 1], self.ti_A[ele_idx, 0, 2], self.ti_A[ele_idx, 0, 3],
                  self.ti_A[ele_idx, 0, 4], self.ti_A[ele_idx, 0, 5], self.ti_A[ele_idx, 0, 6], self.ti_A[ele_idx, 0, 7],
                  self.ti_A[ele_idx, 0, 8], self.ti_A[ele_idx, 0, 9], self.ti_A[ele_idx, 0, 10],
                  self.ti_A[ele_idx, 0, 11]],
                 [self.ti_A[ele_idx, 1, 0], self.ti_A[ele_idx, 1, 1], self.ti_A[ele_idx, 1, 2], self.ti_A[ele_idx, 1, 3],
                  self.ti_A[ele_idx, 1, 4], self.ti_A[ele_idx, 1, 5], self.ti_A[ele_idx, 1, 6], self.ti_A[ele_idx, 1, 7],
                  self.ti_A[ele_idx, 1, 8], self.ti_A[ele_idx, 1, 9], self.ti_A[ele_idx, 1, 10],
                  self.ti_A[ele_idx, 1, 11]],
                 [self.ti_A[ele_idx, 2, 0], self.ti_A[ele_idx, 2, 1], self.ti_A[ele_idx, 2, 2], self.ti_A[ele_idx, 2, 3],
                  self.ti_A[ele_idx, 2, 4], self.ti_A[ele_idx, 2, 5], self.ti_A[ele_idx, 2, 6], self.ti_A[ele_idx, 2, 7],
                  self.ti_A[ele_idx, 2, 8], self.ti_A[ele_idx, 2, 9], self.ti_A[ele_idx, 2, 10],
                  self.ti_A[ele_idx, 2, 11]],
                 [self.ti_A[ele_idx, 3, 0], self.ti_A[ele_idx, 3, 1], self.ti_A[ele_idx, 3, 2], self.ti_A[ele_idx, 3, 3],
                  self.ti_A[ele_idx, 3, 4], self.ti_A[ele_idx, 3, 5], self.ti_A[ele_idx, 3, 6], self.ti_A[ele_idx, 3, 7],
                  self.ti_A[ele_idx, 3, 8], self.ti_A[ele_idx, 3, 9], self.ti_A[ele_idx, 3, 10],
                  self.ti_A[ele_idx, 3, 11]],
                 [self.ti_A[ele_idx, 4, 0], self.ti_A[ele_idx, 4, 1], self.ti_A[ele_idx, 4, 2], self.ti_A[ele_idx, 4, 3],
                  self.ti_A[ele_idx, 4, 4], self.ti_A[ele_idx, 4, 5], self.ti_A[ele_idx, 4, 6], self.ti_A[ele_idx, 4, 7],
                  self.ti_A[ele_idx, 4, 8], self.ti_A[ele_idx, 4, 9], self.ti_A[ele_idx, 4, 10],
                  self.ti_A[ele_idx, 4, 11]],
                 [self.ti_A[ele_idx, 5, 0], self.ti_A[ele_idx, 5, 1], self.ti_A[ele_idx, 5, 2], self.ti_A[ele_idx, 5, 3],
                  self.ti_A[ele_idx, 5, 4], self.ti_A[ele_idx, 5, 5], self.ti_A[ele_idx, 5, 6], self.ti_A[ele_idx, 5, 7],
                  self.ti_A[ele_idx, 5, 8], self.ti_A[ele_idx, 5, 9], self.ti_A[ele_idx, 5, 10],
                  self.ti_A[ele_idx, 5, 11]],
                 [self.ti_A[ele_idx, 6, 0], self.ti_A[ele_idx, 6, 1], self.ti_A[ele_idx, 6, 2], self.ti_A[ele_idx, 6, 3],
                  self.ti_A[ele_idx, 6, 4], self.ti_A[ele_idx, 6, 5], self.ti_A[ele_idx, 6, 6], self.ti_A[ele_idx, 6, 7],
                  self.ti_A[ele_idx, 6, 8], self.ti_A[ele_idx, 6, 9], self.ti_A[ele_idx, 6, 10],
                  self.ti_A[ele_idx, 6, 11]],
                 [self.ti_A[ele_idx, 7, 0], self.ti_A[ele_idx, 7, 1], self.ti_A[ele_idx, 7, 2], self.ti_A[ele_idx, 7, 3],
                  self.ti_A[ele_idx, 7, 4], self.ti_A[ele_idx, 7, 5], self.ti_A[ele_idx, 7, 6], self.ti_A[ele_idx, 7, 7],
                  self.ti_A[ele_idx, 7, 8], self.ti_A[ele_idx, 7, 9], self.ti_A[ele_idx, 7, 10],
                  self.ti_A[ele_idx, 7, 11]],
                 [self.ti_A[ele_idx, 8, 0], self.ti_A[ele_idx, 8, 1], self.ti_A[ele_idx, 8, 2], self.ti_A[ele_idx, 8, 3],
                  self.ti_A[ele_idx, 8, 4], self.ti_A[ele_idx, 8, 5], self.ti_A[ele_idx, 8, 6], self.ti_A[ele_idx, 8, 7],
                  self.ti_A[ele_idx, 8, 8], self.ti_A[ele_idx, 8, 9], self.ti_A[ele_idx, 8, 10],
                  self.ti_A[ele_idx, 8, 11]]
                 ])
            return tmp_mat

    @ti.func
    def compute_Dm(self, i):
        if ti.static(self.dim == 2):
            ia, ib, ic = self.ti_elements[i]
            a, b, c = self.ti_pos_init[ia], self.ti_pos_init[ib], self.ti_pos_init[ic]
            return ti.Matrix.cols([b - a, c - a])
        else:
            idx_a, idx_b, idx_c, idx_d = self.ti_elements[i]
            a, b, c, d = self.ti_pos_init[idx_a], self.ti_pos_init[idx_b], self.ti_pos_init[idx_c], self.ti_pos_init[idx_d]
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
    def precomputation(self, lhs_matrix_np: ti.ext_arr()):
        dimp = self.dim + 1
        # print("Precomputation starts")
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
                # print("ti_elements[e_it]:\n", ti_elements[e_it])

        # Construct A_i matrix for every element / Build A for all the constraints:
        # Strain constraints and area constraints
        # print("A init starts")
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

        # print("Constraints init starts")
        # # Add strain and area/volume constraints to the lhs matrix
        for ele_idx in range(self.n_elements):
            for t in range(2):
                self.fill_idx_vec(ele_idx)

                # May need more considerations:
                for A_row_idx in range(self.dim * (self.dim + 1)):
                    for A_col_idx in range(self.dim * (self.dim + 1)):
                        lhs_row_idx = self.ti_q_idx_vec[ele_idx, A_row_idx]
                        lhs_col_idx = self.ti_q_idx_vec[ele_idx, A_col_idx]
                        for idx in range(self.dim * self.dim):
                            weight = 0.0
                            if t == 0:
                                weight = self.ti_weight_strain[ele_idx]
                            else:
                                weight = self.ti_weight_volume[ele_idx]
                            lhs_matrix_np[lhs_row_idx, lhs_col_idx] += (self.ti_A[ele_idx, idx, A_row_idx] *
                                                                        self.ti_A[ele_idx, idx, A_col_idx] * weight)

        # print("Position constraints starts")
        # Add positional constraints to the lhs matrix
        for i in range(self.n_vertices):
            if self.ti_boundary_labels[i] == 1:
                q_i_x_idx = i * self.dim
                q_i_y_idx = i * self.dim + 1
                lhs_matrix_np[q_i_x_idx, q_i_x_idx] += self.m_weight_positional  # This is the weight of positional constraints
                lhs_matrix_np[q_i_y_idx, q_i_y_idx] += self.m_weight_positional
                if ti.static(self.dim == 3):
                    q_i_z_idx = i * self.dim + 2
                    lhs_matrix_np[q_i_z_idx, q_i_z_idx] += self.m_weight_positional

        # print("lhs matrix starts")
        # Construct lhs matrix without constraints
        for i in range(self.n_vertices):
            for d in ti.static(range(self.dim)):
                lhs_matrix_np[i * self.dim + d, i * self.dim + d] += self.ti_mass[i] / (self.dt * self.dt)
        # print("ti_lhs_mat: after add mass:")
        # print(ti_lhs_matrix[None])

    @ti.func
    def compute_Fi(self, i):
        if ti.static(self.dim == 2):
            ia, ib, ic = self.ti_elements[i]
            a, b, c = self.ti_pos_new[ia], self.ti_pos_new[ib], self.ti_pos_new[ic]
            D_i = ti.Matrix.cols([b - a, c - a])
            return ti.cast(D_i @ self.ti_Dm_inv[i], real)
        else:
            idx_a, idx_b, idx_c, idx_d = self.ti_elements[i]
            a, b, c, d = self.ti_pos_new[idx_a], self.ti_pos_new[idx_b], self.ti_pos_new[idx_c], self.ti_pos_new[idx_d]
            D_i = ti.Matrix.cols([b - a, c - a, d - a])
            return ti.cast(D_i @ self.ti_Dm_inv[i], real)

    @ti.func
    def compute_volume_constraint(self, sigma):
        if ti.static(self.dim == 2):
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
            pos_i = self.ti_pos[vert_idx]
            vel_i = self.ti_vel[vert_idx]
            self.ti_Sn[Sn_idx1] = pos_i[0] + self.dt * vel_i[0] + (self.dt ** 2) * self.ti_ex_force[0][0] / self.ti_mass[
                vert_idx]  # x-direction;
            self.ti_Sn[Sn_idx2] = pos_i[1] + self.dt * vel_i[1] + (self.dt ** 2) * self.ti_ex_force[0][1] / self.ti_mass[
                vert_idx]  # y-direction;
            if ti.static(self.dim == 3):
                Sn_idx3 = vert_idx * self.dim + 2
                self.ti_Sn[Sn_idx3] = pos_i[2] + self.dt * vel_i[2] + (self.dt ** 2) * self.ti_ex_force[0][2] / self.ti_mass[vert_idx]

    @ti.kernel
    def compute_x_xtilde(self):
        if ti.static(self.dim == 2):
            for i in range(self.n_vertices):
                self.x_xtilde[i][0] = self.ti_pos[i][0] - self.ti_Sn[i * self.dim]
                self.x_xtilde[i][1] = self.ti_pos[i][1] - self.ti_Sn[i * self.dim + 1]
        else:
            for i in range(self.n_vertices):
                self.x_xtilde[i][0] = self.ti_pos[i][0] - self.ti_Sn[i * self.dim]
                self.x_xtilde[i][1] = self.ti_pos[i][1] - self.ti_Sn[i * self.dim + 1]
                self.x_xtilde[i][1] = self.ti_pos[i][1] - self.ti_Sn[i * self.dim + 2]
            # print("x-xtilde: ", self.x_xtilde[i], " x: ", self.pos[i], " sn: ", self.Sn[i*2], ", ", self.Sn[i*2+1])

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
                pos_init_i = self.ti_pos_init[i]
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
            self.ti_vel_last[i] = self.ti_vel[i]
            self.ti_vel[i] = (self.ti_pos_new[i] - self.ti_pos[i]) / self.dt
            self.ti_pos_del[i] = self.ti_pos_new[i] - self.ti_pos[i]
            self.ti_pos[i] = self.ti_pos_new[i]

    @ti.kernel
    def warm_up(self):
        for pos_idx in range(self.n_vertices):
            sn_idx1, sn_idx2 = pos_idx * self.dim, pos_idx * self.dim + 1
            self.ti_pos_new[pos_idx][0] = self.ti_Sn[sn_idx1]
            self.ti_pos_new[pos_idx][1] = self.ti_Sn[sn_idx2]
            if ti.static(self.dim == 3):
                sn_idx3 = pos_idx * self.dim + 2
                self.ti_pos_new[pos_idx][2] = self.ti_Sn[sn_idx3]

    @ti.kernel
    def update_pos_new_from_numpy(self, sol: ti.ext_arr()):
        for pos_idx in range(self.n_vertices):
            sol_idx1, sol_idx2 = pos_idx * self.dim, pos_idx * self.dim + 1
            self.ti_pos_new[pos_idx][0] = sol[sol_idx1]
            self.ti_pos_new[pos_idx][1] = sol[sol_idx2]
            if ti.static(self.dim == 3):
                sol_idx3 = pos_idx * self.dim + 2
                self.ti_pos_new[pos_idx][2] = sol[sol_idx3]

    @ti.kernel
    def check_residual(self) -> real:
        residual = 0.0
        for i in range(self.n_vertices):
            residual += (self.ti_last_pos_new[i] - self.ti_pos_new[i]).norm()
            self.ti_last_pos_new[i] = self.ti_pos_new[i]
        # print("residual:", residual)
        return residual

    @ti.kernel
    def copy(self, x: ti.template(), y: ti.template()):
        for i in x:
            y[i] = x[i]

    def build_pos_arr(self, adj_v, arr, rel_pos):
        result = np.zeros((adj_v.shape[0], self.dim))
        result_pos = np.zeros((adj_v.shape[0], self.dim))
        t = 0
        nparr = arr.to_numpy()
        for p in adj_v:  # print("print! ", nparr[p, 0])
            result[t, :] = nparr[p, :]
            result_pos[t, :] = rel_pos[p, :]
            t = t + 1
        return result, result_pos

    def data_one_frame(self, input_p, input_v):
        rhs_np = np.zeros(self.n_vertices * self.dim, np.float64)
        lhs_matrix_np = self.lhs_matrix.to_numpy()
        s_lhs_matrix_np = sparse.csr_matrix(lhs_matrix_np)
        pre_fact_lhs_solve = factorized(s_lhs_matrix_np)
        self.copy(input_p, self.ti_pos)
        self.copy(input_v, self.ti_vel)
        self.build_sn()
        self.warm_up()
        last_record_energy = 100000000000.0
        for itr in range(self.solver_max_iteration):
            self.local_solve_build_bp_for_all_constraints()
            self.build_rhs(rhs_np)
            local_step_energy = self.compute_local_step_energy()
            if local_step_energy > last_record_energy:
                print("Energy Error: LOCAL; Error Amount:",
                      (local_step_energy - last_record_energy) / local_step_energy)
                if (local_step_energy - last_record_energy) / local_step_energy > 0.01:
                    print("Large Error: LOCAL")
            last_record_energy = local_step_energy
            pos_new_np = pre_fact_lhs_solve(rhs_np)
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
        return self.ti_pos_del, self.pos_new, self.vel

    def compute_restT_and_m(self):
        self.ti_mass.fill(0.0)
        self.lhs_matrix.fill(0.0)
        self.precomputation()


    def output_network_data(self, pd_dis, pn_dis, grad_E, frame, T):
        vel = self.ti_vel.to_numpy()
        gradE = grad_E.to_numpy()
        exf = np.array([self.ti_ex_force[0][0], self.ti_ex_force[0][1], self.ti_ex_force[0][2]])
        frame = str(frame).zfill(5)
        if T == 0:
            out_name = "SimData/TrainingData/_" + self.case_info['case_name'] + "_" + str(self.exf_angle1) + "_" + \
                       str(self.exf_angle2) + "_" + str(self.exf_mag) + "_" + frame + ".csv"
        else:
            out_name = "SimData/TestingData/_" + self.case_info['case_name'] + "_" + str(self.exf_angle1) + "_" + \
                       str(self.exf_angle2) + "_" + str(self.exf_mag) + "_" + frame + ".csv"

        ele_count = self.dim + self.dim + self.dim * self.dim + self.dim + self.dim + self.dim + self.dim
        out = np.ones([self.n_vertices, ele_count], dtype=float)
        tt = time.time()
        A_finals = get_local_transformation(self.n_vertices, self.mesh, self.ti_pos.to_numpy(), self.initial_rel_pos,
                                            self.ti_mass.to_numpy(), self.dim)
        ttt = time.time()
        print("solve As: ", ttt - tt)
        i = 0
        for res in A_finals:
            out[i, 0:3] = pd_dis[i, :]  # pd pos
            out[i, 3:6] = pn_dis[i, :]  # pn pos
            out[i, 6:15] = res.get()
            out[i, 15:18] = gradE[i * 3: i * 3 + 3]
            out[i, 18:21] = exf
            out[i, 21:24] = vel[i, :]
            out[i, 24:27] = self.pos_init_out[i, :]
            i = i + 1
        tttt = time.time()
        print("append As: ", tttt - ttt)
        np.savetxt(out_name, out, delimiter=',')

    def output_aux_data(self, f, pn_pos):
        if self.dim == 3:
            name_pd = "SimData/PDAnimSeq/PD_" + self.case_info['case_name'] + "_" + str(self.exf_angle1) + "_" +\
                      str(self.exf_angle2) + "_"+str(self.exf_mag)+"_"+str(f).zfill(6)+".obj"
            name_pn = "SimData/PNAnimSeq/PN_" + self.case_info['case_name'] + "_" + str(self.exf_angle1) + "_" +\
                      str(self.exf_angle2) + "_"+str(self.exf_mag)+"_"+str(f).zfill(6)+".obj"
            output_3d_seq(self.ti_pos.to_numpy(), self.boundary_triangles, name_pd)
            output_3d_seq(pn_pos.to_numpy(), self.boundary_triangles, name_pn)


    def Run(self, pn, is_test, frame_count, scene_info):
        rhs_np = np.zeros(self.n_vertices * self.dim, dtype=np.float64)
        lhs_matrix_np = np.zeros(shape=(self.n_vertices * self.dim, self.n_vertices * self.dim), dtype=np.float64)

        # Init Taichi global variables
        self.init_mesh_DmInv(self.dirichlet, len(self.dirichlet))
        self.precomputation(lhs_matrix_np)
        # lhs_matrix_np = self.ti_lhs_matrix.to_numpy()
        s_lhs_matrix_np = sparse.csr_matrix(lhs_matrix_np)
        pre_fact_lhs_solve = factorized(s_lhs_matrix_np)

        video_manager = ti.VideoManager(output_dir='results/', framerate=24, automatic_build=False)

        if self.dim == 2:
            gui = ti.GUI('2D Simulation Data Generator -- PD -> PN', background_color=0xf7f7f7)
            filename = f'./results/frame_rest.png'
            draw_image(gui, filename, self.ti_pos.to_numpy(), self.mesh_offset, self.mesh_scale, self.ti_elements.to_numpy(), self.n_elements)
        else:
            gui = ti.GUI('3D Simulation Data Generator -- PD -> PN')
            scene_info['model'].set_transform(self.case_info['transformation_mat'])
            update_boundary_mesh(self.ti_pos, scene_info['boundary_pos'], self.case_info)
            scene_info['scene'].input(gui)
            scene_info['tina_mesh'].set_face_verts(scene_info['boundary_pos'])
            scene_info['scene'].render()
            gui.set_image(scene_info['scene'].img)
            gui.show()

        frame_counter = 0
        self.initial_com = calcCenterOfMass(np.arange(self.n_vertices), self.dim,
                                            self.ti_mass.to_numpy(), self.ti_pos.to_numpy())  # this is right
        self.initial_rel_pos = self.ti_pos_init.to_numpy() - self.initial_com
        while frame_counter < frame_count:
            print("//////////////////////////////////////Frame ", frame_counter, "/////////////////////////////////")
            t_0 = time.time()
            self.build_sn()
            self.warm_up()  # Warm up

            pn_start = time.time()
            # get info from pn
            self.copy(self.ti_pos, self.input_xn)
            self.copy(self.ti_vel, self.input_vn)
            pn_dis, _pn_pos, pn_v = pn.data_one_frame(self.input_xn, self.input_vn)
            # print("pn dis \n", self.pn_dis.to_numpy())
            pn_end = time.time()
            print("pn solve time: ", pn_end - pn_start)

            pd_start = time.time()
            for itr in range(self.solver_max_iteration):
                self.local_solve_build_bp_for_all_constraints()
                self.build_rhs(rhs_np)

                pos_new_np = pre_fact_lhs_solve(rhs_np)
                self.update_pos_new_from_numpy(pos_new_np)

                residual = self.check_residual()
                if residual < self.solver_stop_residual:
                    break
            pd_end = time.time()
            print("pd solve time: ", pd_end - pd_start)

            self.update_velocity_pos()      # Update velocity and positions
            self.gradE = pn.get_gradE_from_pd(self.ti_pos)      # get info from pn

            t_out_start = time.time()
            self.output_network_data(self.ti_pos_del.to_numpy(), pn_dis.to_numpy(), self.gradE, frame_counter, is_test)
            t_out_end = time.time()
            print("output time: ", t_out_end - t_out_start)

            t_end = time.time()
            print("whole time for one frame: ", t_end - t_0)
            # Show result
            frame_counter += 1
            if self.dim == 2:
                filename = f'./SimData/TmpRenderedImgs/frame_{frame_counter:05d}.png'
                draw_image(gui, filename, self.ti_pos.to_numpy(), self.mesh_offset, self.mesh_scale,
                           self.ti_elements.to_numpy(), self.n_elements)
            else:
                update_boundary_mesh(self.ti_pos, scene_info['boundary_pos'], self.case_info)
                scene_info['scene'].input(gui)
                scene_info['tina_mesh'].set_face_verts(scene_info['boundary_pos'])
                scene_info['scene'].render()
                gui.set_image(scene_info['scene'].img)
                gui.show()

            self.output_aux_data(frame_counter, _pn_pos)