# from src.Simulators.PN import *
# from src.Simulators.PD import *
# from src.Simulators.PN3D import *
# from src.Simulators.PD3D import *
from src.Simulators.PDSimulator import *
from src.Simulators.PNSimulator import *
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
frame_count = 10

test_case = 1006

# NOTE:
# Please remember to save your data. It will delete all files in xxAnimSeq/ or xxData/ when you exe Run.py.
# New structure will replace the old one after 2-3 versions.

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
    if case_info['dim'] == 2:
        scene_info['gui'] = ti.GUI('2D Simulation Data Generator -- PD -> PN', background_color=0xf7f7f7)
    else:
        import tina
        scene_info['gui'] = ti.GUI('3D Simulation Data Generator -- PD -> PN')
        scene_info['scene'] = tina.Scene(culling=False, clipping=True)
        scene_info['tina_mesh'] = tina.SimpleMesh()
        scene_info['model'] = tina.MeshTransform(scene_info['tina_mesh'])
        scene_info['scene'].add_object(scene_info['model'])
        scene_info['boundary_pos'] = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)

    # Large scale data generation -- 3D
    # sampled_angle1_num = 8
    # sampled_angle2_num = 8
    # sampled_mag_num = 5
    # pd = PDSimulation(case_info, dt)
    # pn = PNSimulation(case_info, dt)
    # pd.set_material(rho, E, nu, dt)
    # pn.set_material(rho, E, nu, dt)
    # for ang_idx1 in range(sampled_angle1_num):
    #     for ang_idx2 in range(sampled_angle2_num):
    #         for mag_idx in range(sampled_mag_num):
    #             pd.initial()
    #             pn.initial()
    #             pn.compute_restT_and_m()
    #             force_info = {'dim': case_info['dim'],
    #                           'force_type': 'dir',
    #                           'exf_angle1': ang_idx1*(180.0 / sampled_angle1_num),
    #                           'exf_angle2': ang_idx2 * (360.0 / sampled_angle2_num),
    #                           'exf_mag': (mag_idx + 5)}
    #             pn.set_force(force_info)
    #             pd.set_force(force_info)
    #             pd.Run(pn, is_test, frame_count, scene_info)

    # Large scale data generation -- 3D -- New structure
    # sampled_angle1_num = 8
    # sampled_angle2_num = 8
    # sampled_mag_num = 5
    # sim_info = {'case_info': case_info, 'dt': dt, 'real': ti.f64}
    # pd = PDSimulation(sim_info)
    # pn = PNSimulation(sim_info)
    # pd.set_material(rho, E, nu)
    # pn.set_material(rho, E, nu)
    # for ang_idx1 in range(sampled_angle1_num):
    #     for ang_idx2 in range(sampled_angle2_num):
    #         for mag_idx in range(sampled_mag_num):
    #             pd.initial()
    #             pn.initial()
    #             pn.compute_restT_and_m()
    #             force_info = {'dim': case_info['dim'],
    #                           'force_type': 'dir',
    #                           'exf_angle1': ang_idx1*(180.0 / sampled_angle1_num),
    #                           'exf_angle2': ang_idx2 * (360.0 / sampled_angle2_num),
    #                           'exf_mag': (mag_idx + 5)}
    #             pn.set_force(force_info)
    #             pd.set_force(force_info)
    #             pd.run(pn, is_test, frame_count, scene_info)

    # Large scale data generation -- 2D
    # sampled_angle_num = 16
    # sampled_mag_num = 9
    # pd = PDSimulation(case_info, dt)
    # pn = PNSimulation(case_info, dt)
    # pd.set_material(rho, E, nu, dt)
    # pn.set_material(rho, E, nu, dt)
    # for angle_idx in range(sampled_angle_num):
    #     for mag_idx in range(sampled_mag_num):
    #         pd.initial()
    #         pn.initial()
    #         pn.compute_restT_and_m()
    #         force_info = {'dim': case_info['dim'],
    #                       'exf_angle': angle_idx * (360.0 / sampled_angle_num),
    #                       'exf_mag': (mag_idx + 1)}
    #         pn.set_force(force_info)
    #         pd.set_force(force_info)
    #         pd.Run(pn, is_test, frame_count, scene_info)

    # Separately generate 2D and 3D
    # pd = PDSimulation(case_info, dt)
    # pn = PNSimulation(case_info, dt)
    # pd.set_material(rho, E, nu, dt)
    # pn.set_material(rho, E, nu, dt)
    # pd.initial()
    # pn.initial()
    # pn.compute_restT_and_m()
    #
    # force_info = {'dim': case_info['dim']}
    # if case_info['dim'] == 2:
    #     force_info['force_type'] = 'dir'
    #     force_info['exf_angle'] = -45.0
    #     force_info['exf_mag'] = 6
    # else:
    #     # 3D direct force field setting
    #     force_info['force_type'] = 'dir'
    #     force_info['exf_angle1'] = 45.0
    #     force_info['exf_angle2'] = 45.0
    #     force_info['exf_mag'] = 10.0
    #
    #     # 3D ring force field setting
    #     # force_info['force_type'] = 'ring'
    #     # force_info['ring_mag'] = 1.0
    #     # force_info['ring_angle'] = 0.0
    #     # force_info['ring_width'] = 0.2
    #
    #     # 3D ring circle force field setting
    #     # force_info['force_type'] = 'ring_circle'
    #     # force_info['ring_mag'] = 1.0
    #     # force_info['ring_angle'] = 0.0
    #     # force_info['ring_width'] = 0.2
    #     # force_info['ring_circle_radius'] = 0.2
    #
    # pn.set_force(force_info)
    # pd.set_force(force_info)
    # pd.Run(pn, is_test, frame_count, scene_info)

    # Test new structure -- Separately generate 2D and 3D
    sim_info = {'case_info': case_info, 'dt': dt, 'real': ti.f64}
    pd = PDSimulation(sim_info)
    pn = PNSimulation(sim_info)
    pd.set_material(rho, E, nu)
    pn.set_material(rho, E, nu)
    pd.initial()
    pn.initial()
    pn.compute_restT_and_m()

    force_info = {'dim': case_info['dim']}
    if case_info['dim'] == 2:
        force_info['force_type'] = 'dir'
        force_info['exf_angle'] = -45.0
        force_info['exf_mag'] = 6
    else:
        # 3D direct force field setting
        force_info['force_type'] = 'dir'
        force_info['exf_angle1'] = 45.0
        force_info['exf_angle2'] = 45.0
        force_info['exf_mag'] = 0.01

        # 3D ring force field setting
        # force_info['force_type'] = 'ring'
        # force_info['ring_mag'] = 1.0
        # force_info['ring_angle'] = 0.0
        # force_info['ring_width'] = 0.2

        # 3D ring circle force field setting
        # force_info['force_type'] = 'ring_circle'
        # force_info['ring_mag'] = 1.0
        # force_info['ring_angle'] = 0.0
        # force_info['ring_width'] = 0.2
        # force_info['ring_circle_radius'] = 0.2
    pd.set_force(force_info)
    pn.set_force(force_info)
    pd.run(pn, is_test, frame_count, scene_info)

