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
from PD2 import*

##############################################################################

mesh, dirichlet, mesh_scale, mesh_offset = read(int(sys.argv[1]))
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

directory = 'output/' + '_'.join(sys.argv) + '/'
os.makedirs(directory + 'images/', exist_ok=True)
print('output directory:', directory)

print("dirichlet: \n", dirichlet)
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

zero = vec()
restT = mat()
vertices = ti.var(ti.i32)
boundary_points = ti.var(ti.i32)
boundary_edges = ti.var(ti.i32)
ti.root.dense(ti.k, n_particles).place(x, xPrev, xTilde, xn, v, m)
ti.root.dense(ti.k, n_particles).place(zero)
ti.root.dense(ti.i, n_elements).place(restT)
ti.root.dense(ti.i, n_elements).place(F)
ti.root.dense(ti.ij, (n_elements, 3)).place(vertices)
ti.root.dense(ti.i, n_boundary_points).place(boundary_points)
ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(boundary_edges)

data_rhs = ti.var(real, shape=2000)
data_mat = ti.var(real, shape=(3, 100000))
data_sol = ti.var(real, shape=2000)
cnt = ti.var(dt=ti.i32, shape=())

n_constraint = 100000
constraints = ti.var(ti.i32, shape=(n_constraint, 3))
cc = ti.var(dt=ti.i32, shape=())
dHat2 = 1e-5
dHat = dHat2 ** 0.5
kappa = 1e4

# external force
exf_angle = np.arange(0,np.pi,20)
exf_mag = np.arange(0,5.0,20)
print("angle: ", exf_angle, " mag: ", exf_mag)
exf_ind = 0
ex_force = ti.sin(exf_angle[exf_ind])*ti.Vector([1.0/math.sqrt(2), 1.0/math.sqrt(2)])*exf_mag[exf_ind]

@ti.func
def compute_T(i):
    a = vertices[i, 0]
    b = vertices[i, 1]
    c = vertices[i, 2]
    ab = x[b] - x[a]
    ac = x[c] - x[a]
    return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])


@ti.kernel
def compute_restT_and_m():
    for i in range(n_elements):
        restT[i] = compute_T(i)
        mass = restT[i].determinant() / 2 * density / 3
        if mass < 0.0:
            print("FATAL ERROR : mesh inverted")
        for d in ti.static(range(3)):
            m[vertices[i, d]] += mass


@ti.kernel
def compute_xn_and_xTilde():
    for i in range(n_particles):
        xn[i] = x[i]
        xTilde[i] = x[i] + dt * v[i]
        xTilde(0)[i] += dt * dt * (ex_force[0]) 
        xTilde(1)[i] += dt * dt * (-9.8 + ex_force[1]) 


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
    print("Search Direction Residual : ", residual / dt)
    return residual


def write_image(f):
    # find_constraints()
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
    for i in range(cc[None]):
        gui.circle(particle_pos[constraints[i, 0]], radius=3, color=0xFF4444)
    gui.show(directory + f'images/{f:06d}.png')


def output_pos(displacement, i):
        outname = "Output_PN/output" + str(i) + ".txt"
        # if not os.path.exists("Output_PN"):
        #     os.makedirs("Output_PN")
        if not os.path.exists(outname):   
            file = open(outname, 'w+')
            file.close()
        np.savetxt(outname, displacement)

def output_pos_pd(displacement, i):
        outname = "Output_PD/output" + str(i) + ".txt"
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

def output_PN(F, disp, frame):
    output_deformation_gradient(F, frame)
    output_pos(disp, frame)

def output_gradE(gradientE, i):
    gradE_sum = 0.0
    for i in gradientE:
        gradE_sum += ti.abs(i)
    print("PD gradE:", gradE_sum)
    outname = "Output_PN/outputR" + str(i) + ".txt"
    if not os.path.exists(outname):
        file = open(outname, 'w+')
        file.close()
    np.savetxt(outname, gradientE)

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

if __name__ == "__main__":
    # set pd simulation
    # pd = PD_Simulation(int(sys.argv[1]), dim)
    # pd.set_Material(density, E, nu, dt)
    # pd.init()

    x.from_numpy(mesh.vertices.astype(np.float32))
    v.fill(0)
    vertices.from_numpy(mesh.faces)
    boundary_points.from_numpy(np.array(list(boundary_points_)))
    boundary_edges.from_numpy(boundary_edges_)
    compute_restT_and_m()
    gui = ti.GUI("MPM", (1024, 1024), background_color=0x112F41)
    vertices_ = vertices.to_numpy()
    zero.fill(0)
    write_image(0)
    total_time = 0.0

    # for output
    if not os.path.exists("Output_PD"):
        os.makedirs("Output_PD")
    if not os.path.exists("Output_PN"):
        os.makedirs("Output_PN")

    # extreme_test()
    
    for forceid in range(1) :
        for f in range(30):
            total_time -= time.time()
            print("==================== Frame: ", f, " ====================")
            compute_xn_and_xTilde()
            # find_constraints()

            # pd to solve:
            # pd.set_Force(forceid)
            # copy(xn, inputxn)
            # copy(v, inputvn)
            # Here PD moves forward one frame;
            # pd_result, pos_result2, pd_pos_new, pd_vel_new = pd.dataoneframe(inputxn, inputvn)
            # output_pos_pd(pos_result2.to_numpy(), f)

            # Compute residual/Grad(E) for the PD in PN:
            data_mat.fill(0)
            data_rhs.fill(0)
            data_sol.fill(0)
            # print("shape of pd_pos_new:", pd_pos_new.shape())
            # print("shape of xn:", xn.shape())
            # print("shape of x:", x.shape())
            # copy(x, temp_x)
            # compute_hessian_and_gradient()
            # compute_pd_gradient()
            # output_gradE(data_rhs.to_numpy()[:n_particles * 2], f)
            # copy(temp_x, x)
            # data_mat.fill(0)
            # data_rhs.fill(0)
            # data_sol.fill(0)

            # pn to solve
            while True:
                data_mat.fill(0)
                data_rhs.fill(0)
                data_sol.fill(0)

                # if residual_save_flag == 0:
                #     # Compute residual/Grad(E) for the PD in PN:
                #     copy(pd_pos_new, xn)
                #     copy(pd_vel_new, v)
                #     compute_hessian_and_gradient()
                #     output_gradE(data_rhs.to_numpy()[:n_particles * 2], f)
                #     copy(inputxn, xn)
                #     copy(inputvn, v)
                #     data_mat.fill(0)
                #     data_rhs.fill(0)
                #     data_sol.fill(0)
                #     residual_save_flag += 1

                compute_hessian_and_gradient()
                data_mat_np = data_mat.to_numpy()
                data_rhs_np = data_rhs.to_numpy()
                data_sol_np = solve_linear_system(data_mat.to_numpy(), data_rhs.to_numpy(), n_particles * 2, dirichlet, zero.to_numpy(), False, cc[None], cnt[None])

                # print("data_mat_np:\n", data_mat_np)
                # print("data_rhs_np:\n", data_rhs_np)
                # print("data_sol_np:\n", data_sol_np)
                data_sol.from_numpy(data_sol_np)
                # data_sol.from_numpy(solve_linear_system(data_mat.to_numpy(), data_rhs.to_numpy(), n_particles * 2, dirichlet, zero.to_numpy(), False, cc[None], cnt[None]))
                if output_residual() < 1e-2 * dt:
                    break

                # print(pd_result.to_numpy())
                # copy_sol(pd_result)
                E0 = compute_energy()
                save_xPrev()
                alpha = 1.0
                apply_sol(alpha)
                E = compute_energy()
                while E > E0:
                    alpha *= 0.5
                    apply_sol(alpha)
                    E = compute_energy()
                    # print(alpha, E0, E)

            # print("alpha: ", alpha)
            compute_v()
            # print("x: \n", x)
            # print("v: \n", v)
            #output the pn result
            # output_PN(F, deltap.to_numpy(), f)
            # output_pos(deltap.to_numpy(), f)

            total_time += time.time()
            print("Time : ", total_time)
            write_image(f + 1)
        # cmd = 'ffmpeg -framerate 12 -i "' + directory + 'images/%6d.png" -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p -threads 20 ' + directory + 'video.mp4'
        # os.system((cmd))
