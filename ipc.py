import taichi as ti

############################################################
@ti.func
def barrier_energy(d, dHat, kappa):
    E = 0.0
    if d < dHat:
        E = -kappa * (d - dHat) * (d - dHat) * ti.log(d / dHat)
    return E


@ti.func
def barrier_gradient(d, dHat, kappa):
    g = 0.0
    t2 = d - dHat
    if d < dHat:
        g = kappa * (t2 * ti.log(d / dHat) * -2.0 - (t2 * t2) / d)
    return g


@ti.func
def barrier_hessian(d, dHat, kappa):
    H = 0.0
    t2 = d - dHat
    if d < dHat:
        H = kappa * ((ti.log(d / dHat) * -2.0 - t2 * 4.0 / d) + 1.0 / (d * d) * (t2 * t2))
    return H


############################################################
@ti.func
def point_point_energy(a, b):
    return (a - b).norm_sqr()


@ti.func
def point_point_gradient(a, b):
    ab2 = 2.0 * (a - b)
    return ti.Matrix([ab2[0], ab2[1], -ab2[0], -ab2[1]])


@ti.func
def point_point_hessian(a, b):
    return ti.Matrix([[2.0, 0.0, -2.0, 0.0], [0.0, 2.0, 0.0, -2.0], [-2.0, 0.0, 2.0, 0.0], [0.0, -2.0, 0.0, 2.0]])


@ti.func
def point_edge_energy(p, e0, e1):
    e = e1 - e0
    numerator = (e[1] * p[0] - e[0] * p[1] + e1[0] * e0[1] - e1[1] * e0[0])
    return numerator * numerator / e.norm_sqr()


@ti.func
def point_edge_gradient(p, e0, e1):
    g = ti.Matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v01, v02, v11, v12, v21, v22 = p[0], p[1], e0[0], e0[1], e1[0], e1[1]
    t13 = -v21 + v11
    t14 = -v22 + v12
    t23 = 1.0 / (t13 * t13 + t14 * t14)
    t25 = ((v11 * v22 + -(v12 * v21)) + t14 * v01) + -(t13 * v02)
    t24 = t23 * t23
    t26 = t25 * t25
    t27 = (v11 * 2.0 + -(v21 * 2.0)) * t24 * t26
    t26 *= (v12 * 2.0 + -(v22 * 2.0)) * t24
    g[0] = t14 * t23 * t25 * 2.0
    g[1] = t13 * t23 * t25 * -2.0
    t24 = t23 * t25
    g[2] = -t27 - t24 * (-v22 + v02) * 2.0
    g[3] = -t26 + t24 * (-v21 + v01) * 2.0
    g[4] = t27 + t24 * (v02 - v12) * 2.0
    g[5] = t26 - t24 * (v01 - v11) * 2.0
    return g


@ti.func
def point_edge_hessian(p, e0, e1):
    H = ti.Matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v01,  v02,  v11,  v12,  v21, v22 = p[0], p[1], e0[0], e0[1], e1[0], e1[1]
    t15 = -v11 + v01
    t16 = -v12 + v02
    t17 = -v21 + v01
    t18 = -v22 + v02
    t19 = -v21 + v11
    t20 = -v22 + v12
    t21 = v11 * 2.0 + -(v21 * 2.0)
    t22 = v12 * 2.0 + -(v22 * 2.0)
    t23 = t19 * t19
    t24 = t20 * t20
    t31 = 1.0 / (t23 + t24)
    t34 = ((v11 * v22 + -(v12 * v21)) + t20 * v01) + -(t19 * v02)
    t32 = t31 * t31
    t33 = ti.pow(t31, 3.0)
    t35 = t34 * t34
    t60 = t31 * t34 * 2.0
    t59 = -(t19 * t20 * t31 * 2.0)
    t62 = t32 * t35 * 2.0
    t64 = t21 * t21 * t33 * t35 * 2.0
    t65 = t22 * t22 * t33 * t35 * 2.0
    t68 = t15 * t21 * t32 * t34 * 2.0
    t71 = t16 * t22 * t32 * t34 * 2.0
    t72 = t17 * t21 * t32 * t34 * 2.0
    t75 = t18 * t22 * t32 * t34 * 2.0
    t76 = t19 * t21 * t32 * t34 * 2.0
    t77 = t20 * t21 * t32 * t34 * 2.0
    t78 = t19 * t22 * t32 * t34 * 2.0
    t79 = t20 * t22 * t32 * t34 * 2.0
    t90 = t21 * t22 * t33 * t35 * 2.0
    t92 = t16 * t20 * t31 * 2.0 + t77
    t94 = -(t17 * t19 * t31 * 2.0) + t78
    t96 = (t18 * t19 * t31 * 2.0 + -t60) + t76
    t99 = (-(t15 * t20 * t31 * 2.0) + -t60) + t79
    t93 = t15 * t19 * t31 * 2.0 + -t78
    t35 = -(t18 * t20 * t31 * 2.0) + -t77
    t97 = (t17 * t20 * t31 * 2.0 + t60) + -t79
    t98 = (-(t16 * t19 * t31 * 2.0) + t60) + -t76
    t100 = ((-(t15 * t16 * t31 * 2.0) + t71) + -t68) + t90
    t19 = ((-(t17 * t18 * t31 * 2.0) + t75) + -t72) + t90
    t102_tmp = t17 * t22 * t32 * t34
    t76 = t15 * t22 * t32 * t34
    t22 = (((-(t15 * t17 * t31 * 2.0) + t62) + -t65) + t76 * 2.0) + t102_tmp * 2.0
    t33 = t18 * t21 * t32 * t34
    t20 = t16 * t21 * t32 * t34
    t79 = (((-(t16 * t18 * t31 * 2.0) + t62) + -t64) + -(t20 * 2.0)) + -(t33 * 2.0)
    t77 = (((t15 * t18 * t31 * 2.0 + t60) + t68) + -t75) + -t90
    t78 = (((t16 * t17 * t31 * 2.0 + -t60) + t72) + -t71) + -t90
    H[0] = t24 * t31 * 2.0
    H[1] = t59
    H[2] = t35
    H[3] = t97
    H[4] = t92
    H[5] = t99
    H[6] = t59
    H[7] = t23 * t31 * 2.0
    H[8] = t96
    H[9] = t94
    H[10] = t98
    H[11] = t93
    H[12] = t35
    H[13] = t96
    t35 = -t62 + t64
    H[14] = (t35 + t18 * t18 * t31 * 2.0) + t33 * 4.0
    H[15] = t19
    H[16] = t79
    H[17] = t77
    H[18] = t97
    H[19] = t94
    H[20] = t19
    t33 = -t62 + t65
    H[21] = (t33 + t17 * t17 * t31 * 2.0) - t102_tmp * 4.0
    H[22] = t78
    H[23] = t22
    H[24] = t92
    H[25] = t98
    H[26] = t79
    H[27] = t78
    H[28] = (t35 + t16 * t16 * t31 * 2.0) + t20 * 4.0
    H[29] = t100
    H[30] = t99
    H[31] = t93
    H[32] = t77
    H[33] = t22
    H[34] = t100
    H[35] = (t33 + t15 * t15 * t31 * 2.0) - t76 * 4.0
    return H


@ti.func
def ipc_overlap(p0, e0, e1):
    e = e1 - e0
    ratio = e.dot(p0 - e0) / e.norm_sqr()
    E = False
    if ratio < 0:
        if point_point_energy(p0, e0) < 1e-8:
            E = True
    elif ratio > 1:
        if point_point_energy(p0, e1) < 1e-8:
            E = True
    else:
        if point_edge_energy(p0, e0, e1) < 1e-8:
            E = True
    if E:
        print("ERIUEIUIDUFIUFDIFU")
    return E


@ti.func
def ipc_energy(p0, e0, e1, dHat2, kappa):
    e = e1 - e0
    ratio = e.dot(p0 - e0) / e.norm_sqr()
    E = 0.0
    if ratio < 0:
        E = barrier_energy(point_point_energy(p0, e0), dHat2, kappa)
    elif ratio > 1:
        E = barrier_energy(point_point_energy(p0, e1), dHat2, kappa)
    else:
        E = barrier_energy(point_edge_energy(p0, e0, e1), dHat2, kappa)
    return E

@ti.func
def ipc_gradient(p0, e0, e1, dHat2, kappa):
    e = e1 - e0
    ratio = e.dot(p0 - e0) / e.norm_sqr()
    g = ti.Matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if ratio < 0:
        g4 = point_point_gradient(p0, e0)
        gb = barrier_gradient(point_point_energy(p0, e0), dHat2, kappa)
        g = ti.Matrix([g4[0], g4[1], g4[2], g4[3], 0.0, 0.0]) * gb
    elif ratio > 1:
        g4 = point_point_gradient(p0, e1)
        gb = barrier_gradient(point_point_energy(p0, e1), dHat2, kappa)
        g = ti.Matrix([g4[0], g4[1], 0.0, 0.0, g4[2], g4[3]]) * gb
    else:
        g6 = point_edge_gradient(p0, e0, e1)
        gb = barrier_gradient(point_edge_energy(p0, e0, e1), dHat2, kappa)
        g = g6 * gb
    return g


@ti.func
def ipc_hessian(p0, e0, e1, dHat2, kappa):
    e = e1 - e0
    ratio = e.dot(p0 - e0) / e.norm_sqr()
    H = ti.Matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    if ratio < 0:
        d2 = point_point_energy(p0, e0)
        gd = point_point_gradient(p0, e0)
        H1 = barrier_gradient(d2, dHat2, kappa) * point_point_hessian(p0, e0)
        H2 = barrier_hessian(d2, dHat2, kappa) * gd @ gd.transpose()
        idx = ti.static([0, 1, 2, 3])
        for i in ti.static(range(4)):
            for j in ti.static(range(4)):
                H[idx[i], idx[j]] += H1[i, j] + H2[i, j]
    elif ratio > 1:
        d2 = point_point_energy(p0, e1)
        gd = point_point_gradient(p0, e1)
        H1 = barrier_gradient(d2, dHat2, kappa) * point_point_hessian(p0, e0)
        H2 = barrier_hessian(d2, dHat2, kappa) * gd @ gd.transpose()
        idx = ti.static([0, 1, 4, 5])
        for i in ti.static(range(4)):
            for j in ti.static(range(4)):
                H[idx[i], idx[j]] += H1[i, j] + H2[i, j]
    else:
        d2 = point_edge_energy(p0, e0, e1)
        gd = point_edge_gradient(p0, e0, e1)
        H1 = barrier_gradient(d2, dHat2, kappa) * point_edge_hessian(p0, e0, e1)
        H2 = barrier_hessian(d2, dHat2, kappa) * gd @ gd.transpose()
        for i in ti.static(range(6)):
            for j in ti.static(range(6)):
                H[i, j] += H1[i * 6 + j] + H2[i, j]
    return H


@ti.func
def point_edge_ccd_broadphase(x0, x1, x2, dHat):
    min_e = ti.min(x1, x2)
    max_e = ti.max(x1, x2)
    return (x0 < max_e + dHat).all() and (min_e - dHat < x0).all()


@ti.func
def moving_point_edge_ccd_broadphase(x0, x1, x2, d0, d1, d2, dHat):
    min_p = ti.min(x0, x0 + d0)
    max_p = ti.max(x0, x0 + d0)
    min_e = ti.min(ti.min(x1, x2), ti.min(x1 + d1, x2 + d2))
    max_e = ti.max(ti.max(x1, x2), ti.max(x1 + d1, x2 + d2))
    return (min_p < max_e + dHat).all() and (min_e - dHat < max_p).all()


@ti.func
def check_overlap(x0, x1, x2, d0, d1, d2, root):
    p0 = x0 + d0 * root
    e0 = x1 + d1 * root
    e1 = x2 + d2 * root
    e = e1 - e0
    ratio = e.dot(p0 - e0) / e.norm_sqr()
    return 0 <= ratio and ratio <= 1


@ti.func
def moving_point_edge_ccd(x0, x1, x2, d0, d1, d2, eta):
    toc = 1.0
    a = d0[0] * (d2[1] - d1[1]) + d0[1] * (d1[0] - d2[0]) + d2[0] * d1[1] - d2[1] * d1[0]
    b = x0[0] * (d2[1] - d1[1]) + d0[0] * (x2[1] - x1[1]) + d0[1] * (x1[0] - x2[0]) + x0[1] * (d1[0] - d2[0]) + d1[1] * x2[0] + d2[0] * x1[1] - d1[0] * x2[1] - d2[1] * x1[0]
    c = x0[0] * (x2[1] - x1[1]) + x0[1] * (x1[0] - x2[0]) + x2[0] * x1[1] - x2[1] * x1[0]
    if a == 0 and b == 0 and c == 0:
        if (x0 - x1).dot(d0 - d1) < 0:
            root = ti.sqrt((x0 - x1).norm_sqr() / (d0 - d1).norm_sqr())
            if root > 0 and root <= 1:
                toc = ti.min(toc, root * (1 - eta))
        if (x0 - x2).dot(d0 - d2) < 0:
            root = ti.sqrt((x0 - x2).norm_sqr() / (d0 - d2).norm_sqr())
            if root > 0 and root <= 1:
                toc = ti.min(toc, root * (1 - eta))
    else:
        if a == 0:
            if b != 0:
                root = -c / b
                if root > 0 and root <= 1:
                    if check_overlap(x0, x1, x2, d0, d1, d2, root):
                        toc = ti.min(toc, root * (1 - eta))
        else:
            delta = b * b - 4 * a * c
            if delta == 0:
                root = -b / (2 * a)
                if root > 0 and root <= 1:
                    if check_overlap(x0, x1, x2, d0, d1, d2, root):
                        toc = ti.min(toc, root * (1 - eta))
            elif delta > 0:
                if b > 0:
                    root = (-b - ti.sqrt(delta)) / (2 * a)
                    if root > 0 and root <= 1:
                        if check_overlap(x0, x1, x2, d0, d1, d2, root):
                            toc = ti.min(toc, root * (1 - eta))
                    root = 2 * c / (-b - ti.sqrt(delta))
                    if root > 0 and root <= 1:
                        if check_overlap(x0, x1, x2, d0, d1, d2, root):
                            toc = ti.min(toc, root * (1 - eta))
                else:
                    root = 2 * c / (-b + ti.sqrt(delta))
                    if root > 0 and root <= 1:
                        if check_overlap(x0, x1, x2, d0, d1, d2, root):
                            toc = ti.min(toc, root * (1 - eta))
                    root = (-b + ti.sqrt(delta)) / (2 * a)
                    if root > 0 and root <= 1:
                        if check_overlap(x0, x1, x2, d0, d1, d2, root):
                            toc = ti.min(toc, root * (1 - eta))
    return toc
