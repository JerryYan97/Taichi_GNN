from src.Simulators.PN import *
from src.Simulators.PD import *

rho = 1e2
E = 1e4
nu = 0.4
dt = 0.01

pd = PDSimulation(1, 2)
pn = PNSimulation(int(1), 2)

running_times = 1
frame_count = 50

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

    for i in range(running_times):
        # pn.generate_exforce()
        # pn.compute_exforce(pn.exf_ind, pn.mag_ind)
        # pd.set_force(pn.exf_ind, pn.mag_ind)
        pn.set_force(0, 3)
        pd.set_force(0, 3)
        # pn.set_force(45, 3)
        # pd.set_force(45, 3)

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
