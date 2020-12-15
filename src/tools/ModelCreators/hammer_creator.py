import matplotlib.pyplot as plt
import numpy as np
import pymesh
import os
import triangle as tr


def hammer_shape():
    pts = np.array([0.0, 0.0])
    pt1 = np.array([-0.7, 0.0])
    pt2 = np.array([-0.7, 0.25])
    pt3 = np.array([0.0, 0.25])
    pt4 = np.array([0.0, 0.65])
    pt5 = np.array([0.25, 0.4])
    pt6 = np.array([0.25, -0.25])
    pt7 = np.array([0.0, -0.25])

    pts = np.vstack((pts, pt1))
    pts = np.vstack((pts, pt2))
    pts = np.vstack((pts, pt3))
    pts = np.vstack((pts, pt4))
    pts = np.vstack((pts, pt5))
    pts = np.vstack((pts, pt6))
    pts = np.vstack((pts, pt7))

    i = np.arange(8)
    seg = np.stack([i, i + 1], axis=1) % 8
    return pts, seg


if __name__ == "__main__":
    pts, seg = hammer_shape()
    A = dict(vertices=pts, segments=seg)
    B = tr.triangulate(A, 'qpa0.0005')

    mesh = pymesh.form_mesh(B['vertices'], B['triangles'])
    pymesh.save_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../../MeshModels/hammer_model_2D.obj", mesh)

    tr.compare(plt, A, B)
    plt.show()

