# from src.Simulators.PN import *
# from src.Simulators.PD import *
from src.Simulators.PN3D import *
from src.Simulators.PD3D import *
from src.Utils.reader import read
import torch
import os
import numpy as np

# ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)
ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)

rho = 1e2
E = 1e4
nu = 0.4
dt = 0.01

running_times = 1
frame_count = 80

test_case = 1
cluster_num = 10

# NOTE: Please remember to save your data. It will delete all files in Outputs/ or Outputs_T/ when you exe Run.py.

# pd->pn
if __name__ == '__main__':
    # Create relevant folders and clean data in the folders:
    is_test = int(input("Data generation mode [0 -- training data /1 -- test data]:"))
    os.makedirs('SimData/PDAnimSeq/', exist_ok=True)
    os.makedirs('SimData/PNAnimSeq/', exist_ok=True)
    os.makedirs('SimData/TmpRenderedImgs/', exist_ok=True)
    for root, dirs, files in os.walk("SimData/PDAnimSeq"):
        for name in files:
            os.remove(os.path.join(root, name))
    for root, dirs, files in os.walk("SimData/PNAnimSeq"):
        for name in files:
            os.remove(os.path.join(root, name))
    if is_test == 0:
        os.makedirs('SimData/TrainingData/', exist_ok=True)
        for root, dirs, files in os.walk("SimData/TrainingData"):
            for name in files:
                os.remove(os.path.join(root, name))
    else:
        os.makedirs('SimData/TestingData/', exist_ok=True)
        for root, dirs, files in os.walk("SimData/TestingData"):
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

    # Separately generate 2D and 3D
    pd = PDSimulation(case_info, dt)
    pn = PNSimulation(case_info, dt)
    pd.set_material(rho, E, nu, dt)
    pn.set_material(rho, E, nu, dt)
    pd.initial()
    pn.initial()
    pn.compute_restT_and_m()

    force_info = {'dim': case_info['dim']}
    if case_info['dim'] == 2:
        force_info['exf_angle'] = -45.0
        force_info['exf_mag'] = 6
    else:
        force_info['exf_angle1'] = 45.0
        force_info['exf_angle2'] = 45.0
        force_info['exf_mag'] = 6.0

    pn.set_force(force_info)
    pd.set_force(force_info)
    pd.Run(pn, is_test, frame_count, scene_info)
