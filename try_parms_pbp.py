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
rho = 1e4
E = 4e5
nu = 0.1
dt = 0.01
running_times = 1
frame_count = 45
test_case = 1006

# pd->pn
if __name__ == '__main__':
    is_test = 0

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

    # Test new structure -- Separately generate 2D and 3D
    whole_start_t = time.time()
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
        # # 3D ring force field setting
        # force_info['force_type'] = 'ring'
        # force_info['ring_mag'] = 0.0075
        # force_info['ring_angle'] = 0.0
        # force_info['ring_width'] = 100.0

        # 3D ring circle force field setting
        # force_info['force_type'] = 'ring_circle'
        # force_info['ring_mag'] = 0.0075
        # force_info['ring_angle'] = 10.0
        # force_info['ring_width'] = 1.0
        # force_info['ring_circle_radius'] = 0.3

        # # 3D point force field setting
        # force_info['force_type'] = 'point'
        # force_info['p_force_box_ind'] = np.array([2.0, 4.0, 2.0])
        # force_info['f'] = np.array([0.0, -1.0, 0.0])
        # force_info['p_mag'] = 1.8

        # 3D point force field setting
        force_info['force_type'] = 'point_by_point'
        force_info['point_ind'] = 661
        force_info['f'] = np.array([-1.0, -1.0, 0.0])
        force_info['p_mag'] = 1.4
        force_info['p_radius'] = 0.1

    pd.set_force(force_info)
    pn.set_force(force_info)
    pd.run(pn, is_test, frame_count, scene_info)

    whole_end_t = time.time()
    print("Whole simulation running time:", whole_end_t - whole_start_t)

