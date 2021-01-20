from src.Simulators.PDSimulator import *
from src.Simulators.PNSimulator import *
from src.Utils.reader import read
import torch
import os, random, time
import numpy as np
# ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)
ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)
rho = 1e4
E = 5e4
nu = 0.1
dt = 0.01
test_case = 1009

# pd->pn
if __name__ == '__main__':
    is_test = 0
    is_test = int(input("Data generation mode [0 -- training data /1 -- test data]:"))
    os.makedirs('SimData/PDAnimSeq/', exist_ok=True)
    os.makedirs('SimData/PNAnimSeq/', exist_ok=True)
    os.makedirs('SimData/TmpRenderedImgs/', exist_ok=True)
    # for root, dirs, files in os.walk("SimData/PDAnimSeq"):
    #     for name in files:
    #         os.remove(os.path.join(root, name))
    # for root, dirs, files in os.walk("SimData/PNAnimSeq"):
    #     for name in files:
    #         os.remove(os.path.join(root, name))
    if is_test == 0:
        os.makedirs('SimData/TrainingData/', exist_ok=True)
    else:
        os.makedirs('SimData/TestingData/', exist_ok=True)

    case_info = read(test_case)
    MESH = case_info["mesh"]
    boundary_points, _, _ = case_info['boundary']
    boundary_points = list(boundary_points)
    # print("n: ", MESH.num_vertices)
    # for i in range(20):
    #     random_num = random.choice(boundary_points)
    #     print(random_num)
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

    for i in range(10):
        choose_p = random.choice(boundary_points)
        pd.initial()
        pn.initial()
        pn.compute_restT_and_m()

        force_info = {'dim': case_info['dim']}
        if case_info['dim'] == 2:
            force_info['force_type'] = 'dir'
            force_info['exf_angle'] = -45.0
            force_info['exf_mag'] = 6
        else:
            # 3D point force field setting
            force_info['force_type'] = 'point_by_point'
            force_info['point_ind'] = choose_p
            force_info['f'] = np.array([-1.0, 0.0, 0.0])
            force_info['p_mag'] = 4.6
            force_info['p_radius'] = 0.1

        pd.set_force(force_info)
        pn.set_force(force_info)
        pd.run_auto_stop(pn, is_test, scene_info)

        whole_end_t = time.time()
        print("Whole simulation running time:", whole_end_t - whole_start_t)

