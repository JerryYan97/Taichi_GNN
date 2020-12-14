# Nov 5 -- 10:48
import sys, os, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import taichi as ti
from Utils.JGSL_WATER import *
from Utils.neo_hookean import *
from Utils.reader import read
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm

##############################################################################

# mesh, dirichlet, mesh_scale, mesh_offset = read(int(sys.argv[1]))
mesh, dirichlet, mesh_scale, mesh_offset = read(2)

print("dirichlet: \n", dirichlet)
##############################################################################

directory = os.getcwd() + '/output/'
video_manager = ti.VideoManager(output_dir=directory + 'images/', framerate=24, automatic_build=False)

ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

real = ti.f64
scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

dim = 2
dt = 0.01
E = 1e4
nu = 0.4
la = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
density = 100

n_particles = mesh.num_vertices
n_elements = mesh.num_faces

x, xPrev, xTilde, xn, v, m = vec(), vec(), vec(), vec(), vec(), scalar()

temp_x = ti.Vector.field(dim, real, n_particles)
inputxn = ti.Vector.field(dim, real, n_particles)
inputvn = ti.Vector.field(dim, real, n_particles)
deltap = ti.Vector.field(dim, real, n_particles)

F = mat()
RR = mat()

zero = vec()
restT = mat()
vertices = ti.var(ti.i32)
ti.root.dense(ti.k, n_particles).place(x, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.k, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.i, n_elements).place(F)
ti.root.dense(ti.i, n_particles).place(RR)
ti.root.dense(ti.ij, (n_elements, 3)).place(vertices)

data_rhs = ti.var(real, shape=2000)
data_mat = ti.var(real, shape=(3, 100000))
data_sol = ti.var(real, shape=2000)

gradE = ti.var(real, shape=2000)
cnt = ti.var(dt=ti.i32, shape=())

# external force -- Angle: from [1, 0] -- counter-clock wise
exf_angle = -45.0
exf_mag = 1
ex_force = ti.Vector.field(dim, real, 1)


@ti.func
def compute_T(i):
    if ti.static(dim == 2):
        ab = x[vertices[i, 1]] - x[vertices[i, 0]]
        ac = x[vertices[i, 2]] - x[vertices[i, 0]]
        return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])


@ti.kernel
def compute_restT_and_m():
    for i in range(n_elements):
        restT[i] = compute_T(i)
        mass = restT[i].determinant() / 2 * density / 3
        # mass = restT[i].determinant() / dim / (dim - 1) * density / (dim + 1)
        if mass < 0.0:
            print("FATAL ERROR : mesh inverted")
        for d in ti.static(range(3)):
            m[vertices[i, d]] += mass


@ti.kernel
def compute_xn_and_xTilde():
    for i in range(n_particles):
        xn[i] = x[i]
        xTilde[i] = x[i] + dt * v[i]
        xTilde(0)[i] += dt * dt * (ex_force[0][0]/m[i])
        xTilde(1)[i] += dt * dt * (ex_force[0][1]/m[i])


@ti.kernel
def compute_hessian_and_gradient():
    cnt[None] = 0
    # inertia
    for i in range(n_particles):
        for d in ti.static(range(2)):
            c = cnt[None] + i * 2 + d
            data_mat[0, c] = i * 2 + d
            data_mat[1, c] = i * 2 + d
            data_mat[2, c] = m[i]
            data_rhs[i * 2 + d] -= m[i] * (x(d)[i] - xTilde(d)[i])
    cnt[None] += n_particles * 2
    # elasticity
    for e in range(n_elements):
        F[e] = compute_T(e) @ restT[e].inverse()
        IB = restT[e].inverse()
        vol0 = restT[e].determinant() / 2
        dPdF = fixed_corotated_first_piola_kirchoff_stress_derivative(F[e], la, mu) * dt * dt * vol0
        # dPdF = neo_hookean_first_piola_kirchoff_stress_derivative(F, la, mu) * dt * dt * vol0
        intermediate = ti.Matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
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
                            vertices[e, 2] * 2, vertices[e, 2] * 2 + 1])
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
        P = fixed_corotated_first_piola_kirchoff_stress(F[e], la, mu) * dt * dt * vol0
        # P = neo_hookean_first_piola_kirchoff_stress(F, la, mu) * dt * dt * vol0
        data_rhs[vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
        data_rhs[vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
        data_rhs[vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
        data_rhs[vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
        data_rhs[vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
        data_rhs[vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]
    cnt[None] += n_elements * 36


@ti.kernel
def save_xPrev():
    for i in range(n_particles):
        xPrev[i] = x[i]


@ti.kernel
def apply_sol(alpha : real):
    for i in range(n_particles):
        for d in ti.static(range(2)):
            x(d)[i] = xPrev(d)[i] + data_sol[i * 2 + d] * alpha


@ti.kernel
def compute_v():
    for i in range(n_particles):
        v[i] = (x[i] - xn[i]) / dt
        deltap[i] = x[i] - xn[i]


@ti.kernel
def output_residual() -> real:
    residual = 0.0
    for i in range(n_particles):
        for d in ti.static(range(2)):
            residual = ti.max(residual, ti.abs(data_sol[i * 2 + d]))
    print("Search Direction Residual : ", residual / dt)
    return residual


@ti.kernel
def compute_energy() -> real:
    total_energy = 0.0
    # inertia
    for i in range(n_particles):
        total_energy += 0.5 * m[i] * (x[i] - xTilde[i]).norm_sqr()
    # elasticity
    for e in range(n_elements):
        tempF = compute_T(e) @ restT[e].inverse()
        vol0 = restT[e].determinant() / 2
        U, sig, V = ti.svd(tempF)
        total_energy += fixed_corotated_energy(ti.Vector([sig[0, 0], sig[1, 1]]), la, mu) * dt * dt * vol0
    return total_energy


def set_exforce():
    x = exf_mag * ti.cos(3.1415926 / 180.0 * exf_angle)
    y = exf_mag * ti.sin(3.1415926 / 180.0 * exf_angle)
    ex_force[0][0], ex_force[0][1] = x, y
    print("ex_force:", ex_force)


def write_image(f):
    particle_pos = (x.to_numpy() + mesh_offset) * mesh_scale
    for i in range(n_elements):
        for j in range(3):
            a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
            gui.line((particle_pos[a][0], particle_pos[a][1]),
                     (particle_pos[b][0], particle_pos[b][1]),
                     radius=1,
                     color=0x4FB99F)
    for i in dirichlet:
        gui.circle(particle_pos[i], radius=3, color=0x44FFFF)
    video_manager.write_frame(gui.get_image())
    gui.show()


if __name__ == "__main__":
    x.from_numpy(mesh.vertices.astype(np.float32))
    v.fill(0)
    vertices.from_numpy(mesh.faces)
    compute_restT_and_m()
    gui = ti.GUI("PN Data Generator", (1024, 1024), background_color=0x112F41)
    vertices_ = vertices.to_numpy()
    zero.fill(0)
    write_image(0)
    total_time = 0.0
    set_exforce()

    for f in range(1, 50):
        total_time -= time.time()
        print("==================== Frame: ", f, " ====================")
        compute_xn_and_xTilde()
        # pn to solve
        while True:
            data_mat.fill(0)
            data_rhs.fill(0)
            data_sol.fill(0)
            compute_hessian_and_gradient()
            data_sol.from_numpy(
                solve_linear_system(data_mat.to_numpy(), data_rhs.to_numpy(), n_particles * 2, dirichlet,
                                    zero.to_numpy(), False, 0, cnt[None]))
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
                print(alpha, E)
        compute_v()
        total_time += time.time()
        write_image(f)
    video_manager.make_video(gif=True, mp4=True)