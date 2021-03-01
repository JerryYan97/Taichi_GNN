from src.Utils.reader import *
from src.Utils.utils_visualization import output_3d_seq, update_boundary_mesh_np
import taichi as ti
import os

# Generate start frame from training data.
# We assume that we have the full training sequences from 0 -- last frame id put under the ./WholeTrainingSet
# Some backup notes:
# 1. File i represents the displacement between i and i+1. E.g. File 0 stores info that can make
# animation goes from frame 0 to the frame 1.
if __name__ == '__main__':
    case_id = 1009
    ti.init(arch=ti.gpu, default_fp=ti.f64)
    if not os.path.isdir("./WholeTrainingSet"):
        raise Exception("Cannot find the whole training set!")
    case_info = read(case_id)
    # Read files
    training_data_files = []
    for _, _, files in os.walk("./WholeTrainingSet"):
        training_data_files.extend(files)
    training_data_files.sort()
    # Check whether input files fulfilling our requirements
    # First file should have index with 0;
    file0_name_len = len(training_data_files[0])
    file0_id_str = training_data_files[0][file0_name_len-9:file0_name_len-4]
    file0_id = int(file0_id_str)
    if file0_id != 0:
        raise Exception("File 0 id should be 0!")
    # Input wanted frame number
    selected_frame_id = int(input("Please input the frame id that you want (Please read note 1):"))
    train_data_files_num = len(training_data_files)
    if selected_frame_id < 0 or selected_frame_id > train_data_files_num + 1:
        raise Exception("Selected frame id is out of range.")
    # Reconstruct animations
    # Assuming our input id is i, we want to construct the ith frame.
    # This means we want to read files from 0 to i-1.
    pd_pos = case_info['mesh'].vertices
    for file_id in range(selected_frame_id - 1):
        displacement = np.genfromtxt("./WholeTrainingSet/" + training_data_files[file_id], delimiter=',')
        displacement = displacement[:, 0:3]
        # pd_pos += displacement
        pd_pos = pd_pos + displacement
        print("processed file id:", file_id)
    output_3d_seq(pd_pos,
                  case_info['boundary'][2],
                  "../../SimData/StartFrame/" + case_info['case_name'] + "_frame_" + f'{selected_frame_id:05}' + ".obj")

    import tina
    scene_info = {}
    scene_info['scene'] = tina.Scene(culling=False, clipping=True)
    scene_info['tina_mesh'] = tina.SimpleMesh()
    scene_info['model'] = tina.MeshTransform(scene_info['tina_mesh'])
    scene_info['scene'].add_object(scene_info['model'])
    scene_info['boundary_pos'] = np.ndarray(shape=(case_info['boundary_tri_num'], 3, 3), dtype=np.float)
    gui = ti.GUI('Test Visualizer')
    update_boundary_mesh_np(pd_pos, scene_info['boundary_pos'], case_info)
    scene_info['scene'].input(gui)
    scene_info['tina_mesh'].set_face_verts(scene_info['boundary_pos'])
    while gui.running:
        scene_info['scene'].render()
        gui.set_image(scene_info['scene'].img)
        gui.show()
