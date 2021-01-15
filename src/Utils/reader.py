import pymesh
import numpy as np
import sys
import os

from .graph_tools import find_boundary
from .utils_visualization import rotate_matrix_y_axis, rotate_general


def fixed_bottom_vert_mp():
    pass


def read(testcase):
    case_info = {}
    if testcase == 1:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/demo3_mesh.obj")
        dirichlet = np.array([i for i in range(11)])
        mesh_scale, mesh_offset = 0.6, 0.4
        case_info['case_name'] = "Bar2d"
        case_info['mesh'] = mesh
        case_info['dim'] = 2
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        return case_info
    elif testcase == 2:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/Sharkey.obj")
        dirichlet = np.array([i for i in range(12)])
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.4
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 1.3
        case_info['case_name'] = "Sharkey2d"
        case_info['mesh'] = mesh
        case_info['dim'] = 2
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        return case_info
    elif testcase == 3:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/p_model_2D.obj")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][0] == mesh.bbox[0][0]:
                dirichlet_list.append(i)
        dirichlet = np.array(dirichlet_list)

        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0
        case_info['case_name'] = "PLetter2d"
        case_info['mesh'] = mesh
        case_info['dim'] = 2
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        return case_info
    elif testcase == 4:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/hammer_model_2D.obj")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][0] == mesh.bbox[0][0]:
                dirichlet_list.append(i)
        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0
        case_info['case_name'] = "Hammer2d"
        case_info['mesh'] = mesh
        case_info['dim'] = 2
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        return case_info
    elif testcase == 1001:
        from tina import translate, scale
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/box3D_v518_t2112.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][2] == mesh.bbox[0][2]:
                dirichlet_list.append(i)
        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0
        center = (mesh.bbox[0] + mesh.bbox[1]) / 2.0
        tmp = mesh.bbox[1] - mesh.bbox[0]
        min_sphere_radius = np.linalg.norm(np.array([tmp[0], tmp[1], tmp[2]])) / 2.0

        case_info['case_name'] = "Beam"
        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['boundary_tri_num'] = len(case_info['boundary'][2])
        case_info['init_translate'] = [0.0, -1.0, 0.0]
        case_info['init_scale'] = 1.0
        case_info['transformation_mat'] = translate([0.0, -1.0, -5.0]) @ rotate_matrix_y_axis(-45.0) @ scale(1.0)
        case_info['center'] = center
        case_info['min_sphere_radius'] = min_sphere_radius
        return case_info
    elif testcase == 1002:
        from tina import translate, scale
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/tet.msh")
        dirichlet_list = [0, 1, 2]
        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)

        case_info['case_name'] = "Tet"
        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['boundary_tri_num'] = len(case_info['boundary'][2])
        case_info['init_translate'] = [0.0, 0.0, 0.0]
        case_info['init_scale'] = 1.0
        case_info['transformation_mat'] = translate([0.0, 0.0, 0.0]) @ rotate_matrix_y_axis(45.0) @ scale(1.0)
        return case_info
    elif testcase == 1003:
        from tina import translate, scale
        # Bunny with bottom fixed
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/bunny3K.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][1] <= mesh.bbox[0][1] + 0.1:
                dirichlet_list.append(i)

        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)
        center = (mesh.bbox[0] + mesh.bbox[1]) / 2.0
        tmp = mesh.bbox[1] - mesh.bbox[0]
        min_sphere_radius = np.linalg.norm(np.array([tmp[0], tmp[1], tmp[2]])) / 2.0

        case_info['case_name'] = "Bunny1"
        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['boundary_tri_num'] = len(case_info['boundary'][2])
        case_info['init_translate'] = [-0.2, -0.5, 0.0]
        case_info['init_scale'] = 1.0
        case_info['transformation_mat'] = translate([-0.3, -0.5, 0.0]) @ rotate_matrix_y_axis(0.0) @ scale(1.0)
        case_info['center'] = center
        case_info['min_sphere_radius'] = min_sphere_radius

        return case_info
    elif testcase == 1004:
        from tina import translate, scale
        # Bunny with 4 fixed points
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/bunny3K.msh")
        dirichlet_list = [665, 777, 1928, 1986]
        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)
        center = (mesh.bbox[0] + mesh.bbox[1]) / 2.0
        tmp = mesh.bbox[1] - mesh.bbox[0]
        min_sphere_radius = np.linalg.norm(np.array([tmp[0], tmp[1], tmp[2]])) / 2.0

        case_info['case_name'] = "Bunny2"
        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['boundary_tri_num'] = len(case_info['boundary'][2])
        case_info['transformation_mat'] = translate([-0.3, -0.5, 0.0]) @ rotate_matrix_y_axis(0.0) @ scale(1.0)
        case_info['center'] = center
        case_info['min_sphere_radius'] = min_sphere_radius

        return case_info
    elif testcase == 1005:
        from tina import translate, scale
        # Dragon with bottom fixed
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/dragon_res4_10019v_38693t.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][1] <= mesh.bbox[0][1] + 0.01:
                dirichlet_list.append(i)

        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)
        center = (mesh.bbox[0] + mesh.bbox[1]) / 2.0
        tmp = mesh.bbox[1] - mesh.bbox[0]
        min_sphere_radius = np.linalg.norm(np.array([tmp[0], tmp[1], tmp[2]])) / 2.0

        case_info['case_name'] = "Dragon"
        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['boundary_tri_num'] = len(case_info['boundary'][2])
        case_info['transformation_mat'] = translate([0.0, -1.0, 0.0]) @ rotate_matrix_y_axis(0.0) @ scale(6.0)
        case_info['center'] = center
        case_info['min_sphere_radius'] = min_sphere_radius

        return case_info
    elif testcase == 1006:
        from tina import translate, scale
        # Dinosaur with bottom fixed
        mesh = pymesh.load_mesh(
            os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/dinosaurlow_16767v_68435t.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][1] <= mesh.bbox[0][1] + 0.01:
                dirichlet_list.append(i)

        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)
        center = (mesh.bbox[0] + mesh.bbox[1]) / 2.0
        tmp = mesh.bbox[1] - mesh.bbox[0]
        min_sphere_radius = np.linalg.norm(np.array([tmp[0], tmp[1], tmp[2]])) / 2.0

        case_info['case_name'] = "dinosaur"
        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['boundary_tri_num'] = len(case_info['boundary'][2])
        case_info['transformation_mat'] = translate([0.0, 0.0, 0.0]) @ rotate_matrix_y_axis(0.0) @ scale(2.0)
        case_info['center'] = center
        case_info['min_sphere_radius'] = min_sphere_radius

        return case_info
    elif testcase == 1007:
        from tina import translate, scale
        # ARM with bottom fixed
        mesh = pymesh.load_mesh(
            os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/armadillolow_17698v_72161t.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][1] <= mesh.bbox[0][1] + 0.01:
                dirichlet_list.append(i)

        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)
        center = (mesh.bbox[0] + mesh.bbox[1]) / 2.0
        tmp = mesh.bbox[1] - mesh.bbox[0]
        min_sphere_radius = np.linalg.norm(np.array([tmp[0], tmp[1], tmp[2]])) / 2.0

        case_info['case_name'] = "armadillo"
        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['boundary_tri_num'] = len(case_info['boundary'][2])
        case_info['transformation_mat'] = translate([0.0, -1.0, 0.0]) @ rotate_matrix_y_axis(0.0) @ scale(1.0)
        case_info['center'] = center
        case_info['min_sphere_radius'] = min_sphere_radius
        return case_info
    elif testcase == 1008:
        from tina import translate, scale
        # Fox with bottom fixed
        mesh = pymesh.load_mesh(
            os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/trans_fox_2036v_7037t.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][1] <= mesh.bbox[0][1] + 0.01:
                dirichlet_list.append(i)

        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)
        center = (mesh.bbox[0] + mesh.bbox[1]) / 2.0
        tmp = mesh.bbox[1] - mesh.bbox[0]
        min_sphere_radius = np.linalg.norm(np.array([tmp[0], tmp[1], tmp[2]])) / 2.0

        case_info['case_name'] = "fox"
        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['boundary_tri_num'] = len(case_info['boundary'][2])
        # case_info['transformation_mat'] = translate([0.0, 0.0, -5.0]) @ rotate_matrix_y_axis(0.0) @ scale(0.1)
        case_info['transformation_mat'] = translate([0.0, -1.0, 0.0]) @ rotate_general(0.0, 0.0, 0.0) @ scale(1.0)
        case_info['center'] = center
        case_info['min_sphere_radius'] = min_sphere_radius
        return case_info
    elif testcase == 1009:
        from tina import translate, scale
        # Low low poly ARM Model
        mesh = pymesh.load_mesh(
            os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/Arm_low2_4713v_17920t.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][1] <= mesh.bbox[0][1] + 0.01:
                dirichlet_list.append(i)

        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)
        center = (mesh.bbox[0] + mesh.bbox[1]) / 2.0
        tmp = mesh.bbox[1] - mesh.bbox[0]
        min_sphere_radius = np.linalg.norm(np.array([tmp[0], tmp[1], tmp[2]])) / 2.0

        case_info['case_name'] = "fox"
        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['boundary_tri_num'] = len(case_info['boundary'][2])
        # case_info['transformation_mat'] = translate([0.0, 0.0, -5.0]) @ rotate_matrix_y_axis(0.0) @ scale(0.1)
        case_info['transformation_mat'] = translate([0.0, -1.0, 0.0]) @ rotate_general(0.0, 180.0, 0.0) @ scale(1.0)
        case_info['center'] = center
        case_info['min_sphere_radius'] = min_sphere_radius
        return case_info
    else:
        raise Exception("Invalid testcase selection.")


def get_routine(v1, v2, v3):
    return [[v1, v2], [v2, v3], [v1, v3]]


def getM(mesh):
    edge_list = []
    for i in range(mesh.faces.shape[0]):
        r = get_routine(mesh.faces[i][0], mesh.faces[i][1], mesh.faces[i][2])
        edge_list.append(r[0])
        edge_list.append(r[1])
        edge_list.append(r[2])
    return edge_list


def deleteDuplicatedElementFromList(list):
    print("sorted list:%s" % list)
    length = len(list)
    lastItem = list[length - 1]
    for i in range(length - 2, -1, -1):
        currentItem = list[i]
        if currentItem == lastItem:
            list.remove(currentItem)
        else:
            lastItem = currentItem
    return list


def get_edge_list(mesh):
    edge_list = getM(mesh)
    edge_list.sort(key=lambda x:x[0], reverse=False)  # 根据第1个元素，sheng序排列
    edge_list = deleteDuplicatedElementFromList(edge_list)
    print("hgjh")
    return edge_list


def read_an_obj(file_name_path):
    mesh = pymesh.load_mesh(file_name_path)
    return mesh.vertices


# bad_vertex = np.logical_not(np.all(np.isfinite(mesh.vertices), axis=1))
# bad_vertex_indices = np.arange(mesh.num_vertices, dtype=int)[bad_vertex]
# for i in bad_vertex_indices:
#     adj_v = mesh.get_vertex_adjacent_vertices(i)
#     adj_v = adj_v[np.logical_not(bad_vertex[adj_v])]


