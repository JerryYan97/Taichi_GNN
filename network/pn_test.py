import sys, os, time
sys.path.insert(0, "../../build")
from JGSL_WATER import *
import taichi as ti
import numpy as np
import pymesh
# from fixed_corotated import *
from neo_hookean import *
from math_tools import *
from ipc import *
from reader import *
from pdfix import*
import scipy as sp

from numpy.linalg import inv
from scipy.linalg import sqrtm

##############################################################################

# mesh, dirichlet, mesh_scale, mesh_offset = read(int(sys.argv[1]))
mesh, dirichlet, mesh_scale, mesh_offset = read(1)
edges = set()
for [i, j, k] in mesh.faces:
    edges.add((i, j))
    edges.add((j, k))
    edges.add((k, i))
boundary_points_ = set()
boundary_edges_ = np.zeros(shape=(0, 2), dtype=np.int32)
for [i, j, k] in mesh.faces:
    if (j, i) not in edges:
        boundary_points_.update([j, i])
        boundary_edges_ = np.vstack((boundary_edges_, [j, i]))
    if (k, j) not in edges:
        boundary_points_.update([k, j])
        boundary_edges_ = np.vstack((boundary_edges_, [k, j]))
    if (i, k) not in edges:
        boundary_points_.update([i, k])
        boundary_edges_ = np.vstack((boundary_edges_, [i, k]))

print("boundary points:", boundary_points_)

##############################################################################
directory = ""
isTest = 2
if isTest is 1:
    directory = os.getcwd() + '/output/' + '_'.join(os.path.basename(sys.argv[0])) + '/'
elif isTest is 0:
    directory = os.getcwd() + '/output/' + '_'.join(os.path.basename(sys.argv[0])+"_ref") + '/'
elif isTest is 2:
    directory = os.getcwd() + '/output/' + '_'.join(os.path.basename(sys.argv[0])+"_pd") + '/'
os.makedirs(directory + 'images/', exist_ok=True)
print('output directory:', directory)
video_manager = ti.VideoManager(output_dir=directory + 'images/', framerate=24, automatic_build=False)

##############################################################################

ti.init(arch=ti.cpu)

real = ti.f32
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
n_boundary_points = len(boundary_points_)
n_boundary_edges = len(boundary_edges_)

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
boundary_points = ti.var(ti.i32)
boundary_edges = ti.var(ti.i32)
ti.root.dense(ti.k, n_particles).place(x, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.k, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.i, n_elements).place(F)
ti.root.dense(ti.i, n_particles).place(RR)
ti.root.dense(ti.ij, (n_elements, 3)).place(vertices)
ti.root.dense(ti.i, n_boundary_points).place(boundary_points)
ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(boundary_edges)

data_rhs = ti.var(real, shape=2000)
data_mat = ti.var(real, shape=(3, 100000))
data_sol = ti.var(real, shape=2000)

gradE = ti.var(real, shape=2000)
cnt = ti.var(dt=ti.i32, shape=())

# external force
exf_angle = np.arange(0, 2*np.pi, 30)
exf_mag = np.arange(0, 10.0, 30)
print("angle: ", exf_angle, " mag: ", exf_mag)
exf_ind = 0
mag_ind = 0
# ex_force = ti.Vector([exf_mag[mag_ind] * math.sin(exf_angle[exf_ind]), exf_mag[mag_ind] * math.cos(exf_angle[exf_ind])])
ex_force = ti.Vector.field(dim, real, 1)
npex_f = np.zeros((2, 1))

# shape matching
initial_com = ti.Vector([0.0, 0.0])
initial_rel_pos = np.array([n_particles, 2])
# shape matching
pi = ti.Vector.field(dim, real, n_particles)
qi = ti.Vector.field(dim, real, n_particles)

def generate_exforce():
    global exf_ind
    global mag_ind
    exf_ind = np.random.randint(30)
    mag_ind = np.random.randint(30)
    # print(exf_ind, " -- ", mag_ind)


@ti.kernel
def compute_exforce(exf_ind: ti.i32, mag_ind: ti.i32):
    x = 0.3*mag_ind * ti.sin(3.1415926/30.0*exf_ind)
    y = 0.3*mag_ind * ti.cos(3.1415926/30.0*exf_ind)
    ex_force[0] = ti.Vector([x, y])


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
        xTilde(1)[i] += dt * dt * (0.0 + ex_force[0][1]/m[i])
        # xTilde(1)[i] -= dt * dt * 9.8 


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
        # total_energy += neo_hookean_energy(ti.Vector([sig[0, 0], sig[1, 1]]), la, mu) * dt * dt * vol0
    return total_energy


@ti.kernel
def compute_pd_gradient():
    cnt[None] = 0
    for i in range(n_particles):
        for d in ti.static(range(2)):
            data_rhs[i * 2 + d] -= m[i] * (x(d)[i] - xTilde(d)[i])
    cnt[None] += n_particles * 2
    for e in range(n_elements):
        F[e] = compute_T(e) @ restT[e].inverse()
        IB = restT[e].inverse()
        vol0 = restT[e].determinant() / 2
        P = fixed_corotated_first_piola_kirchoff_stress(F[e], la, mu) * dt * dt * vol0
        data_rhs[vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
        data_rhs[vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
        data_rhs[vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
        data_rhs[vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
        data_rhs[vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
        data_rhs[vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]
    cnt[None] += n_elements * 36


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


# @ti.kernel
def copy_sol(sol):
    for i in range(n_particles):
        for d in ti.static(range(2)):
            # print("aha", sol[i*2+d])
            data_sol[i*2+d] = sol[i*2+d]


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
    # print("Search Direction Residual : ", residual / dt)
    return residual


def write_image(f):
    if isTest is 0:
        col = 0x4FB99F
    if isTest is 1:
        col = 0xE74C3C
    if isTest is 2:
        col = 0xF4D03F
    particle_pos = (x.to_numpy() + mesh_offset) * mesh_scale
    for i in range(n_elements):
        for j in range(3):
            a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
            gui.line((particle_pos[a][0], particle_pos[a][1]),
                     (particle_pos[b][0], particle_pos[b][1]),
                     radius=1,
                     color=col)
    for i in dirichlet:
        gui.circle(particle_pos[i], radius=3, color=0x44FFFF)
    gui.show(directory + f'images/{f:06d}.png')
    # video_manager.write_frame(gui.get_image())



def output_pos(displacement, i):
        outname = "Output_PN/output"+str(exf_ind)+"_"+str(exf_mag)+"_"+str(i)+".txt"
        # if not os.path.exists("Output_PN"):
        #     os.makedirs("Output_PN")
        if not os.path.exists(outname):   
            file = open(outname, 'w+')
            file.close()
        np.savetxt(outname, displacement)

def output_pos_pd(displacement, i):
    outname = "Output_PD/output"+str(exf_ind)+"_"+str(exf_mag)+"_"+str(i)+".txt"
    if not os.path.exists(outname):   
        file = open(outname, 'w+')
        file.close()
    np.savetxt(outname, displacement)

def output_deformation_gradient(F, i): 
    outname = "Output_PN/outputF" + str(i) + ".txt"
    if not os.path.exists(outname):   
        file = open(outname, 'w+')
        file.close()
    out = np.ones([n_elements, dim*dim], dtype = float)
    for e in range(n_elements): 
        temp = F[e]
        out[e,0] = temp[0,0]
        out[e,1] = temp[0,1] 
        out[e,2] = temp[1,0] 
        out[e,3] = temp[1,1]
    np.savetxt(outname, out)


def output_gradE(gradientE, f):
    gradE_sum = 0.0
    for i in gradientE:
        gradE_sum += ti.abs(i)
    # print("PD gradE:", gradE_sum)
    outname = "Output_PN/outputR"+str(exf_ind)+"_"+str(exf_mag)+"_"+str(f)+".txt"
    if not os.path.exists(outname):
        file = open(outname, 'w+')
        file.close()
    out = np.ones([n_elements, dim], dtype=float)
    for e in range(n_elements): 
        out[e,0] = gradientE[0]
        out[e,1] = gradientE[1] 
    np.savetxt(outname, gradientE)

# @ti.func
def calcCenterOfMass(vind):
    sum = ti.Vector([0.0, 0.0])
    summ = 0.0
    for i in vind:
        for d in ti.static(range(2)):
            sum[d] += m[i] * x[i][d]
        summ += m[i]
    sum[0] /= summ
    sum[1] /= summ
    return sum

@ti.func
def calcA_qq(num_neigh):
    sum = ti.Matrix([[0, 0], [0, 0]])
    for i in range(num_neigh):
        sum += ti.outer_product(qi[i], qi[i].transpose())
    return sum.inverse()

def calcA_pq(p_i, q_i):
    sum = np.zeros((2,2))
    for i in range(p_i.shape[0]):
        sum += np.outer(p_i[i],np.transpose(q_i[i]))
    return sum

# @ti.func
def calcR(A_pq):
    S = sqrtm(np.dot(np.transpose(A_pq),A_pq))
    R = np.dot(A_pq,inv(S))
    return R

def build_pos_arr(adjv, arr, rel_pos):
    result = np.zeros((adjv.shape[0], dim))
    # result = np.array((adjv.shape[0], dim))
    result_pos = np.zeros((adjv.shape[0], dim))
    t = 0
    nparr = arr.to_numpy()
    for p in adjv:
        # print("print! ", nparr[p, 0])
        result[t, 0] = nparr[p, 0]
        result[t, 1] = nparr[p, 1]
        result_pos[t, :] = rel_pos[p, :]
        t = t+1
    return result, result_pos


def get_local_transform(scale, f):
    outname = "Output_PN/outputF" + str(exf_ind) + "_"+str(exf_mag) + "_" + str(f) + ".txt"
    if not os.path.exists(outname):
        file = open(outname, 'w+')
        file.close()
    out = np.ones([n_elements, dim*dim], dtype = float)
    if scale == 1:
        mesh.enable_connectivity()
        for i in range(n_particles):
            adjv = mesh.get_vertex_adjacent_vertices(i)
            adjv = np.append(adjv, i)
            adjv = np.sort(adjv)
            new_pos, init_rel_pos = build_pos_arr(adjv, x, initial_rel_pos)
            com = calcCenterOfMass(adjv).to_numpy()
            curr_rel_pos = np.zeros((adjv.shape[0], dim))
            for j in range(new_pos.shape[0]):
                # print(com)
                # print(curr_rel_pos[j, :])
                # print(new_pos[j, :])
                curr_rel_pos[j, :] = new_pos[j, :] - com
            A_pq = calcA_pq(curr_rel_pos, init_rel_pos)
            R = calcR(A_pq)   
            out[i,0] = R[0,0]
            out[i,1] = R[0,1] 
            out[i,2] = R[1,0]
            out[i,3] = R[1,1]
    np.savetxt(outname, out)

# @ti.kernel
def extreme_test():
    for i in range(11, n_particles):
        for d in ti.static(range(2)):
            pd = np.random.rand()
            x(d)[i] = 0.2 + (pd-0.5)

@ti.kernel
def copy(x: ti.template(), y: ti.template()):
    for i in x:
        y[i] = x[i]


def output_PN(F, disp, frame):
    output_deformation_gradient(F, frame)
    output_pos(disp, frame)
    get_local_transform(1, frame)

def calculate_(F, disp, frame):
    output_deformation_gradient(F, frame)
    output_pos(disp, frame)
    get_local_transform(1, frame)


if __name__ == "__main__":
    testpath = "TestResult"
    realpath = "Outputs_T"

    x.from_numpy(mesh.vertices.astype(np.float32))
    v.fill(0)
    vertices.from_numpy(mesh.faces)
    boundary_points.from_numpy(np.array(list(boundary_points_)))
    boundary_edges.from_numpy(boundary_edges_)
    gui = ti.GUI("MPM_TEST", (1024, 1024), background_color=0x112F41)
    vertices_ = vertices.to_numpy()
    write_image(0)

    Files_Global = []
    Files_Global2 = []
    if isTest == 1:
        for _, _, files in os.walk(testpath):
            Files_Global.append(files)
        Files_Global = Files_Global[0]
        Files_Global.sort()

        for _, _, files in os.walk(realpath):
            Files_Global2.append(files)
        Files_Global2 = Files_Global2[0]
        Files_Global2.sort()

    if isTest is 0 or isTest is 2:
        for _, _, files in os.walk(realpath):
            Files_Global2.append(files)
        Files_Global2 = Files_Global2[0]
        Files_Global2.sort()

    ff = 0
    origin = x.to_numpy().reshape((mesh.num_vertices, 2)).astype(np.float32)

    if isTest == 1:
        for f in range(len(Files_Global)):  # pn from network
            fperframe = np.genfromtxt("{}{}".format(testpath + "/", Files_Global[f]), dtype=np.dtype(str))
            fperframe2 = np.genfromtxt("{}{}".format(realpath + "/", Files_Global2[f]), dtype=np.dtype(str))
            pdpos = fperframe[:, 0:2].astype(float)  # a[start:stop] items start through stop-1
            deltapos = fperframe[:, 2:4].astype(float)  # a[start:stop] items start through stop-1
            pnposr = fperframe2[:, 2:4].astype(float)  # a[start:stop] items start through stop-1
            pnpos = pdpos + deltapos
            pos = pnpos + origin
            origin = pnposr + origin
            x.from_numpy(pos.astype(np.float32))
            write_image(ff + 1)
            video_manager.write_frame(gui.get_image())
            ff = ff + 1

    if isTest == 0:
        for f in range(len(Files_Global2)):  # pn reference
            fperframe = np.genfromtxt("{}{}".format(realpath + "/", Files_Global2[f]), dtype=np.dtype(str))
            pnpos = fperframe[:, 2:4].astype(float)  # a[start:stop] items start through stop-1
            pos = pnpos + origin
            origin = pnpos + origin
            x.from_numpy(pos.astype(np.float32))
            write_image(ff + 1)
            video_manager.write_frame(gui.get_image())
            ff = ff + 1

    if isTest == 2:
        # for f in range(len(Files_Global2)):  # pd reference
        #     fperframe = np.genfromtxt("{}{}".format(realpath + "/", Files_Global2[f]), dtype=np.dtype(str))
        #     pnpos = fperframe[:, 0:2].astype(float)  # a[start:stop] items start through stop-1
        #     pos = pnpos + origin
        #     origin = pnpos + origin
        #     x.from_numpy(pos.astype(np.float32))
        #     write_image(ff + 1)
        #     video_manager.write_frame(gui.get_image())
        #     ff = ff + 1
        for f in range(len(Files_Global2)):  # pd reference
            fperframe = np.genfromtxt("{}{}".format(realpath + "/", Files_Global2[f]), dtype=np.dtype(str))
            pnpos = fperframe[:, 2:4].astype(float)  # a[start:stop] items start through stop-1
            pdpos = fperframe[:, 0:2].astype(float)  # a[start:stop] items start through stop-1
            pos = pdpos + origin
            origin = pnpos + origin
            x.from_numpy(pos.astype(np.float32))
            write_image(ff + 1)
            video_manager.write_frame(gui.get_image())
            ff = ff + 1

    # video_manager.make_video(gif=True, mp4=True)