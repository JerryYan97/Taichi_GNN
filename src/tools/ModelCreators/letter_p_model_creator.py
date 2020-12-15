import matplotlib.pyplot as plt
import numpy as np
import pymesh
import os
import triangle as tr


def half_circle(N, R):
    i = np.arange(N + 1)
    theta = i * np.pi / N + np.pi
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
    seg = np.stack([i, i + 1], axis=1) % (N + 1)
    return pts, seg


def outer_shape():
    pt0 = np.array([-1.4, 0.3])
    pt1 = np.array([-4.2, 0.3])
    pt2 = np.array([-4.2, 1.2])
    pt3 = np.array([1.4, 1.2])
    hc_pts, hc_seg = half_circle(30, 1.4)

    pts = np.vstack((pt0, pt1))
    pts = np.vstack((pts, pt2))
    pts = np.vstack((pts, pt3))
    pts = np.vstack((pts, hc_pts))

    hc_seg += 4
    hc_seg[30][1] = 3
    seg0 = np.array([0, 1])
    seg1 = np.array([1, 2])
    seg2 = np.array([2, 3])
    seg3 = np.array([4, 0])

    seg = np.vstack((seg0, seg1))
    seg = np.vstack((seg, seg2))
    seg = np.vstack((seg, seg3))
    seg = np.vstack((seg, hc_seg))

    return pts, seg


def inner_shape():
    pt0 = np.array([-0.6, 0.3])
    pt1 = np.array([0.6, 0.3])
    hc_pts, hc_seg = half_circle(16, 0.6)

    pts = np.vstack((pt0, pt1))
    pts = np.vstack((pts, hc_pts))

    hc_seg += 2
    hc_seg[16][1] = 1
    seg0 = np.array([0, 1])
    seg1 = np.array([2, 0])
    seg = np.vstack((seg0, seg1))
    seg = np.vstack((seg, hc_seg))

    return pts, seg


if __name__ == "__main__":
    pts, seg = outer_shape()
    inner_pts, inner_seg = inner_shape()

    pts = np.vstack((pts, inner_pts))
    seg = np.vstack((seg, inner_seg + seg.shape[0]))

    A = dict(vertices=pts, segments=seg, holes=[[0.0, -0.1]])
    B = tr.triangulate(A, 'qpa0.02')

    mesh = pymesh.form_mesh(B['vertices'] * 0.3, B['triangles'])
    pymesh.save_mesh(os.path.dirname(os.path.abspath(__file__)) + "/../../../MeshModels/p_model_2D.obj", mesh)

    tr.compare(plt, A, B)
    plt.show()
