import pickle
import sys
import os

if __name__ == '__main__':
    case_id = 1009
    fixed_state_spec = "RightHandFixed"
    dist_arr = pickle.load(open("../../MeshModels/MeshInfo/geodesic_" + str(case_id) + "_" + fixed_state_spec + ".p", "rb"))
    print("Hello World")
