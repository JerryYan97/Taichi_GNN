import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from Utils.reader import read

# In order to make everything work on cluster.
if __name__ == '__main__':
    case_id = 1009
    case_info = read(case_id)
    case_info['elements'] = case_info['mesh'].elements
    case_info['mesh_num_vert'] = case_info['mesh'].num_vertices
    case_info.pop('mesh')
    print("case info before save:", case_info)
    pickle.dump(case_info, open("../../MeshModels/MeshInfo/case_info" + str(case_id) + ".p", "wb"))

    case_info = pickle.load(open("../../MeshModels/MeshInfo/case_info" + str(case_id) + ".p", "rb"))
    print(case_info)

