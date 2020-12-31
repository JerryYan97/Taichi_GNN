import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import taichi as ti
import taichi_three as t3
import numpy as np
import pymesh
from Utils.JGSL_WATER import *
from Utils.neo_hookean import *
from Utils.reader import read
from numpy.linalg import inv
from scipy.linalg import sqrtm
from Utils.utils_visualization import draw_image, set_3D_scene, update_mesh, get_force_field

##############################################################################
case_info = read(1005)
mesh = case_info['mesh']
dirichlet = case_info['dirichlet']
mesh_scale = case_info['mesh_scale']
mesh_offset = case_info['mesh_offset']
dim = case_info['dim']

##############################################################################

ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)

real = ti.f64

dt = 0.01
E = 1e4
nu = 0.4
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
density = 100
n_particles = mesh.num_vertices
if dim == 2:
    n_elements = mesh.num_faces
if dim == 3:
    n_elements = mesh.elements.shape[0]
cnt = ti.field(ti.i32, shape=())

x = ti.Vector.field(dim, real, n_particles)
xPrev = ti.Vector.field(dim, real, n_particles)
xTilde = ti.Vector.field(dim, real, n_particles)
xn = ti.Vector.field(dim, real, n_particles)
v = ti.Vector.field(dim, real, n_particles)
m = ti.field(real, n_particles)

zero = ti.Vector.field(dim, real, n_particles)
restT = ti.Matrix.field(dim, dim, real, n_elements)
vertices = ti.field(ti.i32, (n_elements, dim + 1))

data_rhs = ti.field(real, shape=200000)
data_mat = ti.field(real, shape=(3, 20000000))
data_sol = ti.field(real, shape=200000)

# external force -- Angle: from [1, 0] -- counter-clock wise
ex_force = ti.Vector.field(dim, real, 1)

if dim == 3:
    camera = t3.Camera()
    scene = t3.Scene()
    boundary_points, boundary_edges, boundary_triangles = case_info['boundary']
    model = t3.Model(t3.DynamicMesh(n_faces=len(boundary_triangles) * 2,
                                    n_pos=case_info['mesh'].num_vertices,
                                    n_nrm=len(boundary_triangles) * 2))
    set_3D_scene(scene, camera, model, case_info)


def initial():
    x.from_numpy(mesh.vertices.astype(np.float64))
    if dim == 2:
        vertices.from_numpy(mesh.faces)
    if dim == 3:
        vertices.from_numpy(mesh.elements)
    xPrev.fill(0)
    xTilde.fill(0)
    xn.fill(0)
    v.fill(0)
    m.fill(0)
    zero.fill(0)
    restT.fill(0)
    ################################ external force ######################################
    ex_force.fill(0)

def set_exforce():
    if dim == 2:
        exf_angle = -45.0
        exf_mag = 6
        ex_force[0] = ti.Vector(get_force_field(exf_mag, exf_angle))
    else:
        exf_angle1 = 0.0
        exf_angle2 = 0.0
        exf_mag = 0.0002
        # 1001 6
        # 1003 and 1004 0.06
        # 1005 0.0002
        ex_force[0] = ti.Vector(get_force_field(exf_mag, exf_angle1, exf_angle2, 3))
        print("ex force: ", ex_force[0][0], " ", ex_force[0][1], " ", ex_force[0][2])


@ti.func
def compute_T(i):
    if ti.static(dim == 2):
        ab = x[vertices[i, 1]] - x[vertices[i, 0]]
        ac = x[vertices[i, 2]] - x[vertices[i, 0]]
        return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])
    else:
        ab = x[vertices[i, 1]] - x[vertices[i, 0]]
        ac = x[vertices[i, 2]] - x[vertices[i, 0]]
        ad = x[vertices[i, 3]] - x[vertices[i, 0]]
        return ti.Matrix([[ab[0], ac[0], ad[0]], [ab[1], ac[1], ad[1]], [ab[2], ac[2], ad[2]]])


@ti.kernel
def compute_restT_and_m():
    for i in range(n_elements):
        restT[i] = compute_T(i)
        mass = restT[i].determinant() / dim / (dim - 1) * density / (dim + 1)
        if mass < 0.0:
            print("FATAL ERROR : mesh inverted")
        for d in ti.static(range(dim + 1)):
            m[vertices[i, d]] += mass


@ti.kernel
def compute_xn_and_xTilde():
    if ti.static(dim == 2):
        for i in range(n_particles):
            xn[i] = x[i]
            xTilde[i] = x[i] + dt * v[i]
            xTilde(0)[i] += dt * dt * (ex_force[0][0] / m[i])
            xTilde(1)[i] += dt * dt * (ex_force[0][1] / m[i])
    if ti.static(dim == 3):
        for i in range(n_particles):
            xn[i] = x[i]
            xTilde[i] = x[i] + dt * v[i]
            xTilde(0)[i] += dt * dt * (ex_force[0][0] / m[i])
            xTilde(1)[i] += dt * dt * (ex_force[0][1] / m[i])
            xTilde(2)[i] += dt * dt * (ex_force[0][2] / m[i])


@ti.kernel
def compute_energy() -> real:
    total_energy = 0.0
    # inertia
    for i in range(n_particles):
        total_energy += 0.5 * m[i] * (x[i] - xTilde[i]).norm_sqr()
    # elasticity
    for e in range(n_elements):
        F = compute_T(e) @ restT[e].inverse()
        vol0 = restT[e].determinant() / dim / (dim - 1)
        U, sig, V = svd(F)
        total_energy += fixed_corotated_energy(sig, la, mu) * dt * dt * vol0
    return total_energy


@ti.kernel
def compute_hessian_and_gradient():
    cnt[None] = 0
    # inertia
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            c = cnt[None] + i * dim + d   # 2 * n_p
            data_mat[0, c] = i * dim + d
            data_mat[1, c] = i * dim + d
            data_mat[2, c] = m[i]
            data_rhs[i * dim + d] -= m[i] * (x(d)[i] - xTilde(d)[i])
    cnt[None] += n_particles * dim
    # elasticity
    for e in range(n_elements):
        F = compute_T(e) @ restT[e].inverse()  # 2 * 2
        IB = restT[e].inverse()  # 2 * 2
        vol0 = restT[e].determinant() / dim / (dim - 1)
        dPdF = fixed_corotated_first_piola_kirchoff_stress_derivative(F, la, mu) * dt * dt * vol0   # 4 * 4
        P = fixed_corotated_first_piola_kirchoff_stress(F, la, mu) * dt * dt * vol0  # 2 * 2
        if ti.static(dim == 2):
            intermediate = ti.Matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])  # 6 * 4
            for colI in ti.static(range(4)):
                _000 = dPdF[0, colI] * IB[0, 0]
                _010 = dPdF[0, colI] * IB[1, 0]
                _101 = dPdF[2, colI] * IB[0, 1]
                _111 = dPdF[2, colI] * IB[1, 1]
                _200 = dPdF[1, colI] * IB[0, 0]
                _210 = dPdF[1, colI] * IB[1, 0]
                _301 = dPdF[3, colI] * IB[0, 1]
                _311 = dPdF[3, colI] * IB[1, 1]
                intermediate[2, colI] = _000 + _101
                intermediate[3, colI] = _200 + _301
                intermediate[4, colI] = _010 + _111
                intermediate[5, colI] = _210 + _311
                intermediate[0, colI] = -intermediate[2, colI] - intermediate[4, colI]
                intermediate[1, colI] = -intermediate[3, colI] - intermediate[5, colI]
            indMap = ti.Vector([vertices[e, 0] * 2, vertices[e, 0] * 2 + 1,
                                vertices[e, 1] * 2, vertices[e, 1] * 2 + 1,
                                vertices[e, 2] * 2, vertices[e, 2] * 2 + 1])  # 6
            for colI in ti.static(range(6)):
                _000 = intermediate[colI, 0] * IB[0, 0]
                _010 = intermediate[colI, 0] * IB[1, 0]
                _101 = intermediate[colI, 2] * IB[0, 1]
                _111 = intermediate[colI, 2] * IB[1, 1]
                _200 = intermediate[colI, 1] * IB[0, 0]
                _210 = intermediate[colI, 1] * IB[1, 0]
                _301 = intermediate[colI, 3] * IB[0, 1]
                _311 = intermediate[colI, 3] * IB[1, 1]
                c = cnt[None] + e * 36 + colI * 6 + 0
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[2], indMap[colI], _000 + _101
                c = cnt[None] + e * 36 + colI * 6 + 1
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[3], indMap[colI], _200 + _301
                c = cnt[None] + e * 36 + colI * 6 + 2
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[4], indMap[colI], _010 + _111
                c = cnt[None] + e * 36 + colI * 6 + 3
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[5], indMap[colI], _210 + _311
                c = cnt[None] + e * 36 + colI * 6 + 4
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[0], indMap[colI], - _000 - _101 - _010 - _111
                c = cnt[None] + e * 36 + colI * 6 + 5
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[1], indMap[colI], - _200 - _301 - _210 - _311
            data_rhs[vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
            data_rhs[vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
            data_rhs[vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
            data_rhs[vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
            data_rhs[vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
            data_rhs[vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]
        else:
            Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            intermediate = ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z])
            for colI in ti.static(range(9)):
                intermediate[3, colI] = IB[0, 0] * dPdF[0, colI] + IB[0, 1] * dPdF[3, colI] + IB[0, 2] * dPdF[6, colI]
                intermediate[4, colI] = IB[0, 0] * dPdF[1, colI] + IB[0, 1] * dPdF[4, colI] + IB[0, 2] * dPdF[7, colI]
                intermediate[5, colI] = IB[0, 0] * dPdF[2, colI] + IB[0, 1] * dPdF[5, colI] + IB[0, 2] * dPdF[8, colI]
                intermediate[6, colI] = IB[1, 0] * dPdF[0, colI] + IB[1, 1] * dPdF[3, colI] + IB[1, 2] * dPdF[6, colI]
                intermediate[7, colI] = IB[1, 0] * dPdF[1, colI] + IB[1, 1] * dPdF[4, colI] + IB[1, 2] * dPdF[7, colI]
                intermediate[8, colI] = IB[1, 0] * dPdF[2, colI] + IB[1, 1] * dPdF[5, colI] + IB[1, 2] * dPdF[8, colI]
                intermediate[9, colI] = IB[2, 0] * dPdF[0, colI] + IB[2, 1] * dPdF[3, colI] + IB[2, 2] * dPdF[6, colI]
                intermediate[10, colI] = IB[2, 0] * dPdF[1, colI] + IB[2, 1] * dPdF[4, colI] + IB[2, 2] * dPdF[7, colI]
                intermediate[11, colI] = IB[2, 0] * dPdF[2, colI] + IB[2, 1] * dPdF[5, colI] + IB[2, 2] * dPdF[8, colI]
                intermediate[0, colI] = -intermediate[3, colI] - intermediate[6, colI] - intermediate[9, colI]
                intermediate[1, colI] = -intermediate[4, colI] - intermediate[7, colI] - intermediate[10, colI]
                intermediate[2, colI] = -intermediate[5, colI] - intermediate[8, colI] - intermediate[11, colI]
            indMap = ti.Vector([vertices[e, 0] * 3, vertices[e, 0] * 3 + 1, vertices[e, 0] * 3 + 2,
                                vertices[e, 1] * 3, vertices[e, 1] * 3 + 1, vertices[e, 1] * 3 + 2,
                                vertices[e, 2] * 3, vertices[e, 2] * 3 + 1, vertices[e, 2] * 3 + 2,
                                vertices[e, 3] * 3, vertices[e, 3] * 3 + 1, vertices[e, 3] * 3 + 2])
            for rowI in ti.static(range(12)):
                c = cnt[None] + e * 144 + rowI * 12 + 0
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[3], IB[0, 0] * intermediate[rowI, 0] + IB[0, 1] * intermediate[rowI, 3] + IB[0, 2] * intermediate[rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 1
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[4], IB[0, 0] * intermediate[rowI, 1] + IB[0, 1] * intermediate[rowI, 4] + IB[0, 2] * intermediate[rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 2
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[5], IB[0, 0] * intermediate[rowI, 2] + IB[0, 1] * intermediate[rowI, 5] + IB[0, 2] * intermediate[rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 3
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[6], IB[1, 0] * intermediate[rowI, 0] + IB[1, 1] * intermediate[rowI, 3] + IB[1, 2] * intermediate[rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 4
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[7], IB[1, 0] * intermediate[rowI, 1] + IB[1, 1] * intermediate[rowI, 4] + IB[1, 2] * intermediate[rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 5
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[8], IB[1, 0] * intermediate[rowI, 2] + IB[1, 1] * intermediate[rowI, 5] + IB[1, 2] * intermediate[rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 6
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[9], IB[2, 0] * intermediate[rowI, 0] + IB[2, 1] * intermediate[rowI, 3] + IB[2, 2] * intermediate[rowI, 6]
                c = cnt[None] + e * 144 + rowI * 12 + 7
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[10], IB[2, 0] * intermediate[rowI, 1] + IB[2, 1] * intermediate[rowI, 4] + IB[2, 2] * intermediate[rowI, 7]
                c = cnt[None] + e * 144 + rowI * 12 + 8
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[11], IB[2, 0] * intermediate[rowI, 2] + IB[2, 1] * intermediate[rowI, 5] + IB[2, 2] * intermediate[rowI, 8]
                c = cnt[None] + e * 144 + rowI * 12 + 9
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[0], -data_mat[2, c - 9] - data_mat[2, c - 6] - data_mat[2, c - 3]
                c = cnt[None] + e * 144 + rowI * 12 + 10
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[1], -data_mat[2, c - 9] - data_mat[2, c - 6] - data_mat[2, c - 3]
                c = cnt[None] + e * 144 + rowI * 12 + 11
                data_mat[0, c], data_mat[1, c], data_mat[2, c] = indMap[rowI], indMap[2], -data_mat[2, c - 9] - data_mat[2, c - 6] - data_mat[2, c - 3]
            R10 = IB[0, 0] * P[0, 0] + IB[0, 1] * P[0, 1] + IB[0, 2] * P[0, 2]
            R11 = IB[0, 0] * P[1, 0] + IB[0, 1] * P[1, 1] + IB[0, 2] * P[1, 2]
            R12 = IB[0, 0] * P[2, 0] + IB[0, 1] * P[2, 1] + IB[0, 2] * P[2, 2]
            R20 = IB[1, 0] * P[0, 0] + IB[1, 1] * P[0, 1] + IB[1, 2] * P[0, 2]
            R21 = IB[1, 0] * P[1, 0] + IB[1, 1] * P[1, 1] + IB[1, 2] * P[1, 2]
            R22 = IB[1, 0] * P[2, 0] + IB[1, 1] * P[2, 1] + IB[1, 2] * P[2, 2]
            R30 = IB[2, 0] * P[0, 0] + IB[2, 1] * P[0, 1] + IB[2, 2] * P[0, 2]
            R31 = IB[2, 0] * P[1, 0] + IB[2, 1] * P[1, 1] + IB[2, 2] * P[1, 2]
            R32 = IB[2, 0] * P[2, 0] + IB[2, 1] * P[2, 1] + IB[2, 2] * P[2, 2]
            data_rhs[vertices[e, 1] * 3 + 0] -= R10
            data_rhs[vertices[e, 1] * 3 + 1] -= R11
            data_rhs[vertices[e, 1] * 3 + 2] -= R12
            data_rhs[vertices[e, 2] * 3 + 0] -= R20
            data_rhs[vertices[e, 2] * 3 + 1] -= R21
            data_rhs[vertices[e, 2] * 3 + 2] -= R22
            data_rhs[vertices[e, 3] * 3 + 0] -= R30
            data_rhs[vertices[e, 3] * 3 + 1] -= R31
            data_rhs[vertices[e, 3] * 3 + 2] -= R32
            data_rhs[vertices[e, 0] * 3 + 0] -= -R10 - R20 - R30
            data_rhs[vertices[e, 0] * 3 + 1] -= -R11 - R21 - R31
            data_rhs[vertices[e, 0] * 3 + 2] -= -R12 - R22 - R32
    cnt[None] += n_elements * (dim + 1) * dim * (dim + 1) * dim


@ti.kernel
def save_xPrev():
    for i in range(n_particles):
        xPrev[i] = x[i]


@ti.kernel
def apply_sol(alpha : real):
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            x(d)[i] = xPrev(d)[i] + data_sol[i * dim + d] * alpha


@ti.kernel
def compute_v():
    for i in range(n_particles):
        v[i] = (x[i] - xn[i]) / dt


@ti.kernel
def output_residual() -> real:
    residual = 0.0
    for i in range(n_particles):
        for d in ti.static(range(dim)):
            residual = ti.max(residual, ti.abs(data_sol[i * dim + d]))
    print("Search Direction Residual : ", residual / dt)
    return residual


def write_image(f):
    if dim == 2:
        for i in range(n_elements):
            for j in range(3):
                a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
                gui.line((particle_pos[a][0], particle_pos[a][1]),
                         (particle_pos[b][0], particle_pos[b][1]),
                         radius=1,
                         color=0x4FB99F)
        gui.show(f'output/bunny{f:06d}.png')
    else:
        f = open(f'output/bunny{f:06d}.obj', 'w')
        for i in range(n_particles):
            f.write('v %.6f %.6f %.6f\n' % (particle_pos[i, 0], particle_pos[i, 1], particle_pos[i, 2]))
        for [p0, p1, p2] in boundary_triangles_:
            f.write('f %d %d %d\n' % (p0 + 1, p1 + 1, p2 + 1))
        f.close()


if __name__ == "__main__":
    initial()
    compute_restT_and_m()
    set_exforce()

    video_manager = ti.VideoManager(output_dir=os.getcwd() + '/results/', framerate=24, automatic_build=False)
    frame_counter = 0

    if dim == 2:
        gui = ti.GUI('PN Standalone', background_color=0xf7f7f7)
        filename = f'./results/frame_rest.png'
        draw_image(gui, filename, x.to_numpy(), mesh_offset, mesh_scale, vertices.to_numpy(), n_elements)
    else:
        # filename = f'./results/frame_rest.png'
        gui = ti.GUI('Model Visualizer', camera.res)
        gui.get_event(None)
        model.mesh.pos.from_numpy(case_info['mesh'].vertices.astype(np.float32))
        update_mesh(model.mesh)
        camera.from_mouse(gui)
        scene.render()
        video_manager.write_frame(camera.img)
        gui.set_image(camera.img)
        gui.show()

    for f in range(1000):
        print("==================== Frame: ", f, " ====================")
        compute_xn_and_xTilde()
        while True:
            data_mat.fill(0)
            data_rhs.fill(0)
            data_sol.fill(0)
            compute_hessian_and_gradient()
            if dim == 2:
                data_sol.from_numpy(solve_linear_system(data_mat.to_numpy(), data_rhs.to_numpy(), n_particles * dim,
                                                         np.array(dirichlet), zero.to_numpy(), False, 0, cnt[None]))
            else:
                data_sol.from_numpy(solve_linear_system3(data_mat.to_numpy(), data_rhs.to_numpy(), n_particles * dim,
                                                         np.array(dirichlet), zero.to_numpy(), False, 0, cnt[None]))
            if output_residual() < 1e-4 * dt:
                break
            E0 = compute_energy()
            save_xPrev()
            alpha = 1.0
            apply_sol(alpha)
            E = compute_energy()
            while E > E0:
                alpha *= 0.5
                apply_sol(alpha)
                E = compute_energy()
        compute_v()
        particle_pos = x.to_numpy()
        vertices_ = vertices.to_numpy()
        # write_image(f)
        frame_counter += 1
        filename = f'./results/frame_{frame_counter:05d}.png'
        if dim == 2:
            draw_image(gui, filename, x.to_numpy(), mesh_offset, mesh_scale, vertices.to_numpy(), n_elements)
        else:
            gui.get_event(None)
            model.mesh.pos.from_numpy(x.to_numpy())
            update_mesh(model.mesh)
            camera.from_mouse(gui)
            scene.render()
            video_manager.write_frame(camera.img)
            gui.set_image(camera.img)
            gui.show()