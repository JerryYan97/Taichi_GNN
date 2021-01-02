# from src.Simulators.PN import *
# from src.Simulators.PD import *
from src.Simulators.PN3D import *
from src.Simulators.PD3D import *
from src.Utils.reader import read
import torch
import os

ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False)
# ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)

rho = 1e2
E = 1e4
nu = 0.4
dt = 0.01

running_times = 1
frame_count = 80

test_case = 1001
cluster_num = 10

# NOTE: Please remember to save your data. It will delete all files in Outputs/ or Outputs_T/ when you exe Run.py.

# pd->pn
if __name__ == '__main__':
    # Run settings:
    is_test = 0
    if is_test == 0:
        if not os.path.exists("Outputs"):
            os.makedirs("Outputs")
        for root, dirs, files in os.walk("Outputs/"):
            for name in files:
                os.remove(os.path.join(root, name))

    # Case settings:
    case_info = read(test_case)
    scene_info = {}
    # 3D visualization variables init:
    if case_info['dim'] == 3:
        import tina
        scene_info['scene'] = tina.Scene(culling=False, clipping=True)
        scene_info['tina_mesh'] = tina.SimpleMesh()
        scene_info['model'] = tina.MeshTransform(scene_info['tina_mesh'])
        scene_info['scene'].add_object(scene_info['model'])
        scene_info['boundary_pos'] = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)

    # # ti.init()
    # pd = PDSimulation(test_case, dt)
    # pn = PNSimulation(test_case, dt)
    # sampled_angle1_num = 16
    # sampled_angle2_num = 16
    # angle1_idx = 10
    # angle2_idx = 10
    # mag_idx = 5
    # pn.set_force(angle1_idx*(180.0/sampled_angle1_num), angle2_idx*(360.0/sampled_angle2_num), 0.001*(mag_idx+1))
    # pd.set_force(angle1_idx*(180.0/sampled_angle1_num), angle2_idx*(360.0/sampled_angle2_num), 0.001*(mag_idx+1))
    # pd.set_material(rho, E, nu, dt)
    # pn.set_material(rho, E, nu, dt)
    #
    # pn.initial()
    # pn.compute_restT_and_m()
    #
    # pd.Run(pn, is_test, frame_count)

    #TODO: ti.init wrong???
    # pn = PNSimulation(test_case, 0.01)
    # sampled_angle1_num = 16
    # sampled_angle2_num = 16
    # angle1_idx = 10
    # angle2_idx = 10
    # mag_idx = 5
    # pn.set_force(angle1_idx * (180.0 / sampled_angle1_num), angle2_idx * (360.0 / sampled_angle2_num),
    #              0.001 * (mag_idx + 1))
    # pn.set_material(rho, E, nu, dt)
    # pn.initial()
    # pn.Run2()

    # TODO: generate large size of data
    # is_test = int(input("Data generation mode [0 -- training data /1 -- test data]:"))
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
    # sampled_angle1_num = 8
    # sampled_angle2_num = 8
    # sampled_mag_num = 5
    # pd = PDSimulation(test_case, dt)
    # pn = PNSimulation(test_case, dt)
    # pd.set_material(rho, E, nu, dt)
    # pn.set_material(rho, E, nu, dt)
    # pd.initial_scene()
    # for ang_idx1 in range(sampled_angle1_num):
    #     for ang_idx2 in range(sampled_angle2_num):
    #         for mag_idx in range(sampled_mag_num):
    #             pd.initial()
    #             pn.initial()
    #             pn.compute_restT_and_m()
    #
    #             pn.set_force(ang_idx1*(180.0 / sampled_angle1_num), ang_idx2 * (360.0 / sampled_angle2_num), (mag_idx + 5))
    #             pd.set_force(ang_idx1*(180.0 / sampled_angle1_num), ang_idx2 * (360.0 / sampled_angle2_num), (mag_idx + 5))
    #             pd.Run(pn, is_test, frame_count)


    # foor loop generate
    # pd = PDSimulation(test_case, dt)
    # pn = PNSimulation(test_case, dt)
    # pd.set_material(rho, E, nu, dt)
    # pn.set_material(rho, E, nu, dt)
    #
    # for ang_idx1 in range(1):
    #     for ang_idx2 in range(2):
    #         for mag_idx in range(1):
    #             # ti.reset()
    #             #
    #             # pd = PDSimulation(test_case, dt)
    #             # pn = PNSimulation(test_case, dt)
    #             pd.initial()
    #             pn.initial()
    #             pn.compute_restT_and_m()
    #
    #             pn.set_force(ang_idx1*(180.0 / sampled_angle1_num), ang_idx2 * (360.0 / sampled_angle2_num), (mag_idx + 1))
    #             pd.set_force(ang_idx1*(180.0 / sampled_angle1_num), ang_idx2 * (360.0 / sampled_angle2_num), (mag_idx + 1))
    #
    #             pd.Run(pn, is_test, frame_count)

    # Separately generate
    pd = PDSimulation(case_info, dt)
    pn = PNSimulation(case_info, dt)
    pd.set_material(rho, E, nu, dt)
    pn.set_material(rho, E, nu, dt)
    pd.initial()
    pn.initial()
    pn.compute_restT_and_m()
    pn.set_force(0, 0, 6)
    pd.set_force(0, 0, 6)
    pd.Run(pn, is_test, frame_count, scene_info)


    # pd2 = PDSimulation(test_case, dt)
    # pn2 = PNSimulation(test_case, dt)
    # pd2.set_material(rho, E, nu, dt)
    # pn2.set_material(rho, E, nu, dt)
    # pd2.initial()
    # pn2.initial()
    # pn2.compute_restT_and_m()
    # pn2.set_force(0 * (180.0 / sampled_angle1_num), 1 * (360.0 / sampled_angle2_num), (0 + 1))
    # pd2.set_force(0 * (180.0 / sampled_angle1_num), 1 * (360.0 / sampled_angle2_num), (0 + 1))
    # pd2.Run(pn2, is_test, frame_count)


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
