import meshio, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
import numpy as np
from decimal import Decimal

def transferFromNode(points, cells, name):
    mesh = meshio.Mesh(points, cells)
    meshio.write(
        "../../MeshModels/" + name + ".msh",  # str, os.PathLike, or buffer/ open file
        mesh,
        # file_format="vtk",  # optional if first argument is a path; inferred from extension
    )

def readNodeFile(pointfilepath, cellfilepath):
    f = open(pointfilepath)
    line = f.readline()
    line = line.split()
    num_vertices = int(line[0])
    dim = int(line[1])
    points = np.zeros((num_vertices, dim))
    t = 0
    while t < num_vertices:
        line = f.readline()
        line = line.split()
        for i in range(dim):
            points[t, i] = Decimal(line[i])
        t = t + 1
    f.close()

    f = open(cellfilepath)
    line = f.readline()
    line = line.split()
    num_element = int(line[0])
    dim = int(line[1])
    cells = np.zeros((num_element, dim)).astype(int)
    t = 0
    # if dim == 2:
    #     c_type = 'Triangle'
    # if dim == 3:
    #     c_type = 'tetra'
    while t < num_element:
        line = f.readline()
        line = line.split()
        for i in range(dim):
            cells[t, i] = int(line[i])
        t = t + 1
    final_cells = {'tetra': cells}
    f.close()

    return points, final_cells


def readNodeFile2(pointfilepath, cellfilepath):
    f = open(pointfilepath)
    line = f.readline()
    line = line.split()
    num_vertices = int(line[0])
    dim = int(line[1])
    points = np.zeros((num_vertices, dim))
    t = 0
    while t < num_vertices:
        line = f.readline()
        line = line.split()
        for i in range(dim):
            points[t, i] = Decimal(line[i+1])
        t = t + 1
    f.close()

    f = open(cellfilepath)
    line = f.readline()
    line = line.split()
    num_element = int(line[0])
    dim = int(line[1])
    cells = np.zeros((num_element, dim)).astype(int)
    t = 0
    while t < num_element:
        line = f.readline()
        line = line.split()
        for i in range(dim):
            cells[t, i] = int(line[i+1])
        t = t + 1
    final_cells = {'tetra': cells}
    f.close()

    return points, final_cells

if __name__ == '__main__':
    # p, c = readNodeFile("../../MeshModels/bunny.node", "../../MeshModels/bunny.ele")
    # transferFromNode(p, c)

    # someng wrong with dragon
    p, c = readNodeFile2("../../MeshModels/dragon.node", "../../MeshModels/dragon.ele")
    transferFromNode(p, c, "dragon")

    # mesh = meshio.read("../../MeshModels/box3D_v518_t2112.msh")
    # print(mesh.points)
    # print(mesh.cells)


