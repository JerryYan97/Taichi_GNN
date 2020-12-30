# from src.Simulators.PN import *
# from src.Simulators.PD import *
from src.Simulators.PN3D import *
from src.Simulators.PD3D import *
from src.Utils.reader import read
from src.Utils.utils_gcn import K_means
import torch
import os
ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)

rho = 1e2
E = 1e4
nu = 0.4
dt = 0.01

running_times = 1
frame_count = 1

test_case = 1004
cluster_num = 10

# NOTE: Please remember to save your data. It will delete all files in Outputs/ or Outputs_T/ when you exe Run.py.

# pd->pn
if __name__ == '__main__':
    # simple data generation
    is_test = 0
    if is_test == 0:
        if not os.path.exists("Outputs"):
            os.makedirs("Outputs")
        for root, dirs, files in os.walk("Outputs/"):
            for name in files:
                os.remove(os.path.join(root, name))

    ti.init()
    pd = PDSimulation(test_case, 0.01)
    pn = PNSimulation(test_case, 0.01)
    sampled_angle1_num = 16
    sampled_angle2_num = 16
    angle1_idx = 10
    angle2_idx = 10
    mag_idx = 5
    pn.set_force(angle1_idx*(180.0/sampled_angle1_num), angle2_idx*(360.0/sampled_angle2_num), 0.002*(mag_idx+1))
    pd.set_force(angle1_idx*(180.0/sampled_angle1_num), angle2_idx*(360.0/sampled_angle2_num), 0.002*(mag_idx+1))
    pd.set_material(rho, E, nu, dt)
    pn.set_material(rho, E, nu, dt)
    pn.initial()
    pn.compute_restT_and_m()
    pd.Run(pn, is_test, frame_count)

    # pn = PNSimulation(test_case, 0.01)
    # pn.Run2()

    # is_test = int(input("Data generation mode [0 -- training data /1 -- test data]:"))
    #
    # if is_test == 0:
    #     if not os.path.exists("Outputs"):
    #         os.makedirs("Outputs")
    #     for root, dirs, files in os.walk("Outputs/"):
    #         for name in files:
    #             os.remove(os.path.join(root, name))
    # else:
    #     if not os.path.exists("Outputs_T"):
    #         os.makedirs("Outputs_T")
    #     for root, dirs, files in os.walk("Outputs_T/"):
    #         for name in files:
    #             os.remove(os.path.join(root, name))

    # Large scale data generation
    # sampled_angle_num = 16
    # sampled_mag_num = 9
    #
    # for angle_idx in range(sampled_angle_num):
    #     for mag_idx in range(sampled_mag_num):
    #         ti.reset()
    #         pd = PDSimulation(test_case, 2)
    #         pn = PNSimulation(test_case, 2)
    #
    #         pn.set_force(angle_idx * (360.0 / sampled_angle_num), (mag_idx + 1))
    #         pd.set_force(angle_idx * (360.0 / sampled_angle_num), (mag_idx + 1))
    #
    #         pd.set_material(rho, E, nu, dt)
    #         pn.set_material(rho, E, nu, dt)
    #         pn.compute_restT_and_m()
    #         pn.zero.fill(0)
    #         pd.Run(pn, is_test, frame_count)

    # for i in range(running_times):
    #     pd = PDSimulation(test_case, 2)
    #     pn = PNSimulation(int(test_case), 2)
    #     # pn.set_force(25, 6)
    #     # pd.set_force(25, 6)
    #     pn.set_force(12.3, 6.6)
    #     pd.set_force(12.3, 6.6)
    #
    #     pd.set_material(rho, E, nu, dt)
    #     pn.set_material(rho, E, nu, dt)
    #     pn.compute_restT_and_m()
    #     pn.zero.fill(0)
    #     pd.Run(pn, is_test, frame_count)

# pn->pd
# if __name__ == '__main__':
#     for i in range(running_times):
#         # pn.generate_exforce()
#         # pn.compute_exforce(pn.exf_ind, pn.mag_ind)
#         pn.set_force(10, 12)
#         pd.set_force(10, 12)
#
#         pd.set_material(rho, E, nu, dt)
#         pn.set_material(rho, E, nu, dt)
#         pd.compute_restT_and_m()
#         pn.Run(pd, is_test, frame_count)
