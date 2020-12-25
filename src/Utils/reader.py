import pymesh
import numpy as np
import sys
import os
import taichi_three as t3

from .graph_tools import find_boundary


def read(testcase):
    case_info = {}
    if testcase == 1:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/demo3_mesh.obj")
        dirichlet = np.array([i for i in range(11)])
        mesh_scale, mesh_offset = 0.6, 0.4
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
        case_info['mesh'] = mesh
        case_info['dim'] = 2
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        return case_info
    elif testcase == 1001:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/box3D_v518_t2112.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][2] == mesh.bbox[0][2]:
                dirichlet_list.append(i)
        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['init_transformation'] = t3.transform(t3.rotateY(45.0), [-0.2, -1.1, -2.0])
        case_info['light_dir'] = [-0.2, -0.6, 0.0]
        return case_info
    elif testcase == 1002:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/tet.msh")
        dirichlet_list = [0, 1, 2]
        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)

        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['init_transformation'] = t3.transform(t3.rotateY(0.0), [0.0, 0.0, 0.0])
        case_info['light_dir'] = [-0.8, -0.6, -1.0]
        return case_info
    elif testcase == 1003:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/bunny3K.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][1] <= mesh.bbox[0][1] + 0.1:
                dirichlet_list.append(i)

        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)

        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['init_transformation'] = t3.transform(t3.rotateY(0.0), [-0.5, -1.0, 0.0])
        case_info['light_dir'] = [-0.8, -0.6, -1.0]
        return case_info
    elif testcase == 1004:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/dragon_res4_10019v_38693t.msh")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][1] <= mesh.bbox[0][1] + 0.01:
                dirichlet_list.append(i)

        dirichlet = np.array(dirichlet_list)
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0

        print("mesh elements:", mesh.elements)

        case_info['mesh'] = mesh
        case_info['dim'] = 3
        case_info['dirichlet'] = dirichlet
        case_info['mesh_scale'] = mesh_scale
        case_info['mesh_offset'] = mesh_offset
        case_info['boundary'] = find_boundary(mesh.elements)
        case_info['init_transformation'] = t3.transform(t3.rotateY(-75.0), [0.0, -0.2, 2.5])
        case_info['light_dir'] = [-0.8, -0.6, -1.0]
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

# mesh, _, _, _ = read(1)

def get_edge_list(mesh):
    edge_list = getM(mesh)
    edge_list.sort(key=lambda x:x[0], reverse=False)  # 根据第1个元素，sheng序排列
    edge_list = deleteDuplicatedElementFromList(edge_list)
    print("hgjh")
    return edge_list


# bad_vertex = np.logical_not(np.all(np.isfinite(mesh.vertices), axis=1))
# bad_vertex_indices = np.arange(mesh.num_vertices, dtype=int)[bad_vertex]
# for i in bad_vertex_indices:
#     adj_v = mesh.get_vertex_adjacent_vertices(i)
#     adj_v = adj_v[np.logical_not(bad_vertex[adj_v])]


