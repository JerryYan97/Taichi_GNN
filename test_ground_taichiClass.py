from abc import ABC, abstractmethod
import taichi as ti


@ti.data_oriented
class SimulatorBase(ABC):
    def __init__(self, case_info, sim_info):
        self.case_info = case_info
        self.n_vertices = sim_info['n_vertices']
        self.dt = sim_info['dt']
        self.rho = sim_info['rho']
        self.E = 1.0
        self.nu = 1.0
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.ti_ex_force = ti.Vector.field(self.case_info['dim'], ti.f32, self.n_vertices)

    def set_material(self, rho, ym, nu, dt):
        self.dt = dt
        self.rho = rho
        self.E = ym
        self.nu = nu
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @ti.kernel
    def _set_dir_force(self, dir_force: ti.template()):
        for i in range(self.n_vertices):
            self.ti_ex_force[i] = dir_force

    def set_dir_force(self):
        self._set_dir_force(ti.Vector([1.0, 1.0, 1.0]))
        # print(self.ti_ex_force)

    @abstractmethod
    def initial(self):
        pass


# @ti.data_oriented
class PDSimulation(SimulatorBase):
    def __init__(self, case_info, sim_info):
        super().__init__(case_info, sim_info)

    @ti.kernel
    def _show_force(self):
        for i in range(self.n_vertices):
            self.ti_ex_force[i] += ti.Vector([1.0, 1.0, 1.0])

    def show_force(self):
        self._show_force()
        print(self.ti_ex_force)

    def initial(self):
        print("PDSimulation")


if __name__ == "__main__":
    case_info = {'dim': 3}
    sim_info = {'n_vertices': 100, 'dt': 0.01, 'rho': 100}
    pd_sim = PDSimulation(case_info, sim_info)
    # pd_sim.initial()
    pd_sim.set_dir_force()
    pd_sim.show_force()

# import tina
#
#
# ti.svd()

# class Count(object):
#     def __init__(self, min, max):
#         self.min = min
#         self.max = max
#         self.current = None
#
#     def __getattribute__(self, item):
#         print(type(item), item)
#         if item.startswith('cur'):
#             raise AttributeError
#         return object.__getattribute__(self, item)
#         # or you can use ---return super().__getattribute__(item)
#
#
# obj1 = Count(1, 10)
# print(obj1.min)
# print(obj1.max)
# print(obj1.current)