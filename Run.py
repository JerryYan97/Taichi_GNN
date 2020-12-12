from src.Simulators.PN import *
from src.Simulators.PD import *
from src.Utils.reader import read
from src.Utils.utils_gcn import K_means
import torch

ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)

rho = 1e2
E = 1e4
nu = 0.4
dt = 0.01

running_times = 1
frame_count = 50

test_case = 1
cluster_num = 10

# NOTE: Please remember to save your data. It will delete all files in Outputs/ or Outputs_T/ when you exe Run.py.

# pd->pn
if __name__ == '__main__':

    is_test = int(input("Data generation mode [0 -- training data /1 -- test data]:"))

    if is_test == 0:
        if not os.path.exists("Outputs"):
            os.makedirs("Outputs")
        for root, dirs, files in os.walk("Outputs/"):
            for name in files:
                os.remove(os.path.join(root, name))
    else:
        if not os.path.exists("Outputs_T"):
            os.makedirs("Outputs_T")
        for root, dirs, files in os.walk("Outputs_T/"):
            for name in files:
                os.remove(os.path.join(root, name))

    # Generate cluster
    if not os.path.exists("Saved_Cluster"):
        os.makedirs("Saved_Cluster")
    for root, dirs, files in os.walk("Saved_Cluster/"):
        for name in files:
            os.remove(os.path.join(root, name))

    mesh, _, _, _ = read(int(test_case))
    _, child_list, parent_list, belonging = K_means(mesh, cluster_num)
    cluster = np.zeros(len(mesh.vertices) + 1, dtype=int)
    for i in parent_list:
        cluster[i] = i
    for i in range(len(child_list)):
        cluster[child_list[i]] = belonging[i]
    cluster[len(mesh.vertices)] = cluster_num
    np.savetxt("Saved_Cluster/cluster.csv", cluster, delimiter=',', fmt='%d')

    # Large scale data generation
    # sampled_angle_num = 16
    # sampled_mag_num = 8
    #
    # for angle_idx in range(sampled_angle_num):
    #     for mag_idx in range(sampled_mag_num):
    #         ti.reset()
    #         pd = PDSimulation(1, 2)
    #         pn = PNSimulation(int(1), 2)
    #
    #         pn.set_force(angle_idx * (360.0 / sampled_angle_num), (mag_idx + 1))
    #         pd.set_force(angle_idx * (360.0 / sampled_angle_num), (mag_idx + 1))
    #
    #         pd.set_material(rho, E, nu, dt)
    #         pn.set_material(rho, E, nu, dt)
    #         pn.compute_restT_and_m()
    #         pn.zero.fill(0)
    #         pd.Run(pn, is_test, frame_count)

    # Debug test
    # for i in range(2):
    #     ti.reset()
    #     pd = PDSimulation(1, 2)
    #     pn = PNSimulation(int(1), 2)
    #     # pn.set_force(0, 3)
    #     # pd.set_force(0, 3)
    #     #
    #     # pd.set_material(rho, E, nu, dt)
    #     # pn.set_material(rho, E, nu, dt)
    #     # pn.compute_restT_and_m()
    #     # pn.zero.fill(0)
    #     # pd.Run(pn, is_test, frame_count)
    #
    #     pn.set_force(0, 8)
    #     pd.set_force(0, 8)
    #
    #     pd.set_material(rho, E, nu, dt)
    #     pn.set_material(rho, E, nu, dt)
    #     pn.compute_restT_and_m()
    #     pn.zero.fill(0)
    #     pd.Run(pn, is_test, frame_count)


    for i in range(running_times):
        pd = PDSimulation(test_case, 2)
        pn = PNSimulation(int(test_case), 2)
        # pn.generate_exforce()
        # pn.compute_exforce(pn.exf_ind, pn.mag_ind)
        # pd.set_force(pn.exf_ind, pn.mag_ind)
        # pn.set_force(-45, 3)
        # pd.set_force(-45, 3)
        pn.set_force(12.3, 6.6)
        pd.set_force(12.3, 6.6)

        pd.set_material(rho, E, nu, dt)
        pn.set_material(rho, E, nu, dt)
        pn.compute_restT_and_m()
        pn.zero.fill(0)
        pd.Run(pn, is_test, frame_count)

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
