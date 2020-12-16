import pymesh
import numpy as np
import sys
import os

def read(testcase):
    if testcase == 1:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/demo3_mesh.obj")
        dirichlet = np.array([i for i in range(11)])
        mesh_scale= 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.6
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 0.9
        shesh_scale, mesh_offset = 0.6, 0.4
        return mesh, dirichlet, shesh_scale, mesh_offset
    elif testcase == 2:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/Sharkey.obj")
        dirichlet = np.array([i for i in range(12)])
        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.4
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 1.3
        return mesh, dirichlet, mesh_scale, mesh_offset
    elif testcase == 3:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/p_model_2D.obj")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][0] == mesh.bbox[0][0]:
                dirichlet_list.append(i)
        dirichlet = np.array(dirichlet_list)

        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0
        return mesh, dirichlet, mesh_scale, mesh_offset
    elif testcase == 4:
        mesh = pymesh.load_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../MeshModels/hammer_model_2D.obj")
        dirichlet_list = []
        for i in range(mesh.num_vertices):
            if mesh.vertices[i][0] == mesh.bbox[0][0]:
                dirichlet_list.append(i)
        dirichlet = np.array(dirichlet_list)

        mesh_scale = 1 / (np.amax(mesh.vertices) - np.amin(mesh.vertices)) * 0.3
        mesh_offset = -(np.amax(mesh.vertices) + np.amin(mesh.vertices)) / 2 + 2.0
        return mesh, dirichlet, mesh_scale, mesh_offset
    else:
        assert True, "Invalid testcase selection."


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


