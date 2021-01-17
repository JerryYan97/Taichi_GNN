from abc import ABC, abstractmethod
import sys, os
import taichi as ti
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.utils_visualization import get_force_field, get_ring_force_field, \
    get_ring_circle_force_field, get_point_force_field, get_point_force_field_by_point


@ti.data_oriented
class SimulatorBase(ABC):
    def __init__(self, sim_info):
        self.case_info = sim_info['case_info']
        self.dt = sim_info['dt']
        self.real = sim_info['real']
        # Mesh
        self.dim = self.case_info['dim']
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
            self.boundary_points, self.boundary_edges, self.boundary_triangles = self.case_info['boundary']

        # Simulator Field
        self.ti_x = ti.Vector.field(self.dim, self.real, self.n_vertices)
        self.ti_mass = ti.field(self.real, self.n_vertices)
        self.ti_vel = ti.Vector.field(self.dim, self.real, self.n_vertices)
        self.ti_elements = ti.Vector.field(self.dim + 1, int, self.n_elements)

        # Materials and parameters
        self.rho = 100
        self.E = 1e4
        self.nu = 0.4
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.p_ind = 0

        # External force
        self.ti_ex_force = ti.Vector.field(self.dim, self.real, self.n_vertices)
        self.exf_mag = 0.0
        self.force_type = 'None'
        if self.dim == 2:
            self.exf_angle = 0.0
        else:
            self.center = self.case_info['center']

            self.ring_mag = 1.0
            self.ring_angle = 0.0
            self.ring_width = 0.2
            self.ring_circle_radius = 0.2
            self.ti_center = ti.Vector([self.center[0], self.center[1], self.center[2]])
            self.exf_angle1 = 0.0
            self.exf_angle2 = 0.0
            self.min_sphere_radius = self.case_info['min_sphere_radius']

            self.pf_mag = 0.0
            self.pf_bbox_ind = np.array([0.0, 0.0, 0.0])
            self.pf_force = np.array([0.0, 0.0, 0.0])
            self.b_min = self.case_info['bbox_min']
            self.b_dx = self.case_info['bbox_dx']
            self.ti_b_min = ti.Vector([self.b_min[0], self.b_min[1], self.b_min[2]])
            self.ti_b_max = ti.Vector([self.b_min[0], self.b_min[1], self.b_min[2]])
            self.ti_pf_force = ti.Vector([self.pf_force[0], self.pf_force[1], self.pf_force[2]])

            self.pf_ind = -1
            self.pf_radius = 0.0

            self.ti_A = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def set_dir_force(self, dir_force: ti.template()):  # only need to set once
        for i in range(self.n_vertices):
            self.ti_ex_force[i] = dir_force

    @ti.kernel
    def set_ring_force_3D(self):    # set every time
        for i in range(self.n_vertices):
            self.ti_ex_force[i] = get_ring_force_field(self.ring_mag, self.ring_width,
                                                       self.ti_center, self.ti_x[i],
                                                       self.ring_angle, 3)

    @ti.kernel
    def set_ring_circle_force_3D(self):     # set every time
        for i in range(self.n_vertices):
            self.ti_ex_force[i] = get_ring_circle_force_field(self.ring_mag, self.ring_width,
                                                              self.ti_center, self.ti_x[i], self.ring_angle,
                                                              self.ring_circle_radius * self.min_sphere_radius, 3)

    @ti.kernel
    def set_point_force_3D(self):   # set only one time
        for i in range(self.n_vertices):
            self.ti_ex_force[i] = get_point_force_field(self.ti_b_min, self.ti_b_max, self.ti_x[i], self.ti_pf_force)

    @ti.kernel
    def set_point_force_by_point_3D(self):   # set only one time
        for i in range(self.n_vertices):  # t_pos, pos, radius, force
            self.ti_ex_force[i] = get_point_force_field_by_point(self.ti_x[self.pf_ind], self.ti_x[i],
                                                                 self.pf_radius, self.ti_pf_force)

    @ti.kernel
    def copy(self, x: ti.template(), y: ti.template()):
        for i in x:
            y[i] = x[i]

    def update_force_field(self):
        if self.dim == 3:
            if self.force_type == 'ring':
                self.set_ring_force_3D()
            elif self.force_type == 'ring_circle':
                self.set_ring_circle_force_3D()

    def set_force(self, force_info):
        if force_info['dim'] != self.case_info['dim']:
            raise AttributeError("Input force dim is not equal to the simulator's dim!")
        self.force_type = force_info['force_type']
        if force_info['force_type'] == 'dir':
            if force_info['dim'] == 2:
                self.exf_angle = force_info['exf_angle']
                self.exf_mag = force_info['exf_mag']
                dir_force = ti.Vector(get_force_field(self.exf_mag, self.exf_angle))
            else:
                self.exf_angle1 = force_info['exf_angle1']
                self.exf_angle2 = force_info['exf_angle2']
                self.exf_mag = force_info['exf_mag']
                dir_force = ti.Vector(get_force_field(self.exf_mag, self.exf_angle1, self.exf_angle2, 3))
            self.set_dir_force(dir_force)
        elif force_info['force_type'] == 'ring':
            if force_info['dim'] == 2:
                raise TypeError("Dim 2 doesn't have ring force field.")
            else:
                self.ring_mag = force_info['ring_mag']
                self.ring_width = force_info['ring_width']
                self.ring_angle = force_info['ring_angle']
        elif force_info['force_type'] == 'ring_circle':
            if force_info['dim'] == 2:
                raise TypeError("Dim 2 doesn't have ring force field.")
            else:
                self.ring_mag = force_info['ring_mag']
                self.ring_width = force_info['ring_width']
                self.ring_angle = force_info['ring_angle']
                self.ring_circle_radius = force_info['ring_circle_radius']
        elif force_info['force_type'] == 'point':
            if force_info['dim'] == 2:
                raise TypeError("Dim 2 doesn't have point force field.")
            else:
                self.pf_bbox_ind = force_info['p_force_box_ind']  # vec3
                self.pf_force = force_info['f']
                self.pf_mag = force_info['p_mag']
                self.ti_b_min = ti.Vector([self.b_min[0] + self.b_dx[0] * (self.pf_bbox_ind[0]-1.0),
                                           self.b_min[1] + self.b_dx[1] * (self.pf_bbox_ind[1]-1.0),
                                           self.b_min[2] + self.b_dx[2] * (self.pf_bbox_ind[2]-1.0)])
                self.ti_b_max = ti.Vector([self.b_min[0]+self.b_dx[0]*self.pf_bbox_ind[0],
                                           self.b_min[1]+self.b_dx[1]*self.pf_bbox_ind[1],
                                           self.b_min[2]+self.b_dx[2]*self.pf_bbox_ind[2]])
                self.ti_pf_force = self.pf_mag * ti.Vector([self.pf_force[0], self.pf_force[1], self.pf_force[2]])
                print("print min: ", self.ti_b_min[0], self.ti_b_min[1], self.ti_b_min[2])
                print("print max: ", self.ti_b_max[0], self.ti_b_max[1], self.ti_b_max[2])
        elif force_info['force_type'] == 'point_by_point':
            if force_info['dim'] == 2:
                raise TypeError("Dim 2 doesn't have point by point force field.")
            else:
                self.pf_ind = force_info['point_ind']  # vec3
                self.pf_force = force_info['f']
                self.pf_mag = force_info['p_mag']
                self.pf_radius = force_info['p_radius']
                self.ti_pf_force = self.pf_mag * ti.Vector([self.pf_force[0], self.pf_force[1], self.pf_force[2]])
                print("print min: ", self.ti_b_min[0], self.ti_b_min[1], self.ti_b_min[2])
                print("print max: ", self.ti_b_max[0], self.ti_b_max[1], self.ti_b_max[2])
        else:
            raise TypeError("The input force type is invalid")

    def base_initial(self):
        if self.dim == 2:
            self.ti_elements.from_numpy(self.mesh.faces)
        else:
            self.ti_elements.from_numpy(self.mesh.elements)
        self.ti_x.from_numpy(self.mesh.vertices)
        self.ti_mass.fill(0)
        self.ti_vel.fill(0)
        self.ti_ex_force.fill(0)

    def set_material(self, rho, ym, nu):
        self.rho = rho
        self.E = ym
        self.nu = nu
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @abstractmethod
    def initial(self):
        pass

