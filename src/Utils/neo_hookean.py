import taichi as ti
from .math_tools import *


@ti.func
def fixed_corotated_energy(sig: ti.template(), la, mu):
    if ti.static(sig.n == 2):
        sigma = ti.Matrix.zero(ti.get_runtime().default_fp, sig.n, 1)
        for i in ti.static(range(sig.n)):
            sigma[i] = sig[i, 0 if ti.static(sig.m == 1) else i]
        sigmam12Sum = (sigma - ti.Vector([1, 1])).norm_sqr()
        sigmaProdm1 = sigma[0] * sigma[1] - 1
        return mu * sigmam12Sum + la / 2 * sigmaProdm1 * sigmaProdm1
    else:
        sigma = ti.Matrix.zero(ti.get_runtime().default_fp, sig.n, 1)
        for i in ti.static(range(sig.n)):
            sigma[i] = sig[i, 0 if ti.static(sig.m == 1) else i]
        sigmam12Sum = (sigma - ti.Vector([1, 1, 1])).norm_sqr()
        sigmaProdm1 = sigma[0] * sigma[1] * sigma[2] - 1
        return mu * sigmam12Sum + la / 2 * sigmaProdm1 * sigmaProdm1


@ti.func
def elasticity_gradient(sig: ti.template(), la, mu):
    if ti.static(sig.n == 2):
        sigma = ti.Matrix.zero(ti.get_runtime().default_fp, sig.n, 1)
        for i in ti.static(range(sig.n)):
            sigma[i] = sig[i, 0 if ti.static(sig.m == 1) else i]
        sigmaProdm1lambda = la * (sigma[0] * sigma[1] - 1)
        sigmaProd_noI = ti.Vector([sigma[1], sigma[0]])
        _2u = mu * 2
        return ti.Vector([_2u * (sigma[0] - 1) + sigmaProd_noI[0] * sigmaProdm1lambda,
                          _2u * (sigma[1] - 1) + sigmaProd_noI[1] * sigmaProdm1lambda])
    else:
        sigma = ti.Matrix.zero(ti.get_runtime().default_fp, sig.n, 1)
        for i in ti.static(range(sig.n)):
            sigma[i] = sig[i, 0 if ti.static(sig.m == 1) else i]
        sigmaProdm1lambda = la * (sigma[0] * sigma[1] * sigma[2] - 1)
        sigmaProd_noI = ti.Vector([sigma[1] * sigma[2], sigma[2] * sigma[0], sigma[0] * sigma[1]])
        _2u = mu * 2
        return ti.Vector([_2u * (sigma[0] - 1) + sigmaProd_noI[0] * sigmaProdm1lambda,
                          _2u * (sigma[1] - 1) + sigmaProd_noI[1] * sigmaProdm1lambda,
                          _2u * (sigma[2] - 1) + sigmaProd_noI[2] * sigmaProdm1lambda])


@ti.func
def elasticity_hessian(sig: ti.template(), la, mu):
    if ti.static(sig.n == 2):
        sigma = ti.Matrix.zero(ti.get_runtime().default_fp, sig.n, 1)
        for i in ti.static(range(sig.n)):
            sigma[i] = sig[i, 0 if ti.static(sig.m == 1) else i]
        sigmaProd = sigma[0] * sigma[1]
        sigmaProd_noI = ti.Vector([sigma[1], sigma[0]])
        _2u = mu * 2
        return ti.Matrix([[_2u + la * sigmaProd_noI[0] * sigmaProd_noI[0],
                           la * ((sigmaProd - 1) + sigmaProd_noI[0] * sigmaProd_noI[1])],
                          [la * ((sigmaProd - 1) + sigmaProd_noI[0] * sigmaProd_noI[1]),
                           _2u + la * sigmaProd_noI[1] * sigmaProd_noI[1]]])
    else:
        sigma = ti.Matrix.zero(ti.get_runtime().default_fp, sig.n, 1)
        for i in ti.static(range(sig.n)):
            sigma[i] = sig[i, 0 if ti.static(sig.m == 1) else i]
        sigmaProd = sigma[0] * sigma[1] * sigma[2]
        sigmaProd_noI = ti.Vector([sigma[1] * sigma[2], sigma[2] * sigma[0], sigma[0] * sigma[1]])
        _2u = mu * 2
        H01 = la * (sigma[2] * (sigmaProd - 1) + sigmaProd_noI[0] * sigmaProd_noI[1])
        H02 = la * (sigma[1] * (sigmaProd - 1) + sigmaProd_noI[0] * sigmaProd_noI[2])
        H12 = la * (sigma[0] * (sigmaProd - 1) + sigmaProd_noI[1] * sigmaProd_noI[2])
        return ti.Matrix([[_2u + la * sigmaProd_noI[0] * sigmaProd_noI[0], H01, H02],
                          [H01, _2u + la * sigmaProd_noI[1] * sigmaProd_noI[1], H12],
                          [H02, H12, _2u + la * sigmaProd_noI[2] * sigmaProd_noI[2]]])


@ti.func
def fixed_corotated_first_piola_kirchoff_stress(F, la, mu):
    J = F.determinant()
    JFinvT = cofactor(F)
    U, sig, V = svd(F)
    # U, sig, V = my_svd(F)
    R = U @ V.transpose()
    return 2 * mu * (F - R) + la * (J - 1) * JFinvT


@ti.func
def fixed_corotated_first_piola_kirchoff_stress_derivative(F, la, mu, dPdF, M_field, U_field, V_field, ele_idx, dt, vol0):
    if ti.static(F.n == 2):
        U, sig, V = svd(F)
        # U, sig, V = my_svd(F)
        for row_idx in ti.static(range(2)):
            for col_idx in ti.static(range(2)):
                U_field[ele_idx, row_idx, col_idx] = U[row_idx, col_idx]
                V_field[ele_idx, row_idx, col_idx] = V[row_idx, col_idx]

        sigma = ti.Vector([sig[0, 0], sig[1, 1]])
        dE_div_dsigma = elasticity_gradient(sigma, la, mu)
        d2E_div_dsigma2 = project_pd(elasticity_hessian(sigma, la, mu))

        leftCoef = mu - la / 2 * (sigma[0] * sigma[1] - 1)
        rightCoef = dE_div_dsigma[0] + dE_div_dsigma[1]
        sum_sigma = ti.max(sigma[0] + sigma[1], 0.000001)
        rightCoef /= (2 * sum_sigma)
        B = project_pd(ti.Matrix([[leftCoef + rightCoef, leftCoef - rightCoef], [leftCoef - rightCoef, leftCoef + rightCoef]]))

        M_field[ele_idx, 0, 0] = d2E_div_dsigma2[0, 0]
        M_field[ele_idx, 0, 3] = d2E_div_dsigma2[0, 1]
        M_field[ele_idx, 1, 1] = B[0, 0]
        M_field[ele_idx, 1, 2] = B[0, 1]
        M_field[ele_idx, 2, 1] = B[1, 0]
        M_field[ele_idx, 2, 2] = B[1, 1]
        M_field[ele_idx, 3, 0] = d2E_div_dsigma2[1, 0]
        M_field[ele_idx, 3, 3] = d2E_div_dsigma2[1, 1]
        for j in range(2):
            for i in range(2):
                for s in range(2):
                    for r in range(2):
                        ij = ti.static(j * 2 + i)
                        rs = ti.static(s * 2 + r)
                        dPdF_tmp = M_field[ele_idx, 0, 0] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 0] * U_field[ele_idx, r, 0] * V_field[ele_idx, s, 0] + M_field[ele_idx, 0, 3] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 0] * \
                                       U_field[ele_idx, r, 1] * V_field[ele_idx, s, 1] + M_field[ele_idx, 1, 1] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 1] * U_field[ele_idx, r, 0] * V_field[ele_idx, s, 1] + M_field[ele_idx, 1, 2] * \
                                       U_field[ele_idx, i, 0] * V_field[ele_idx, j, 1] * U_field[ele_idx, r, 1] * V_field[ele_idx, s, 0] + M_field[ele_idx, 2, 1] * U_field[ele_idx, i, 1] * V_field[ele_idx, j, 0] * U_field[ele_idx, r, 0] * \
                                       V_field[ele_idx, s, 1] + M_field[ele_idx, 2, 2] * U_field[ele_idx, i, 1] * V_field[ele_idx, j, 0] * U_field[ele_idx, r, 1] * V_field[ele_idx, s, 0] + M_field[ele_idx, 3, 0] * U_field[ele_idx, i, 1] * \
                                       V_field[ele_idx, j, 1] * U_field[ele_idx, r, 0] * V_field[ele_idx, s, 0] + M_field[ele_idx, 3, 3] * U_field[ele_idx, i, 1] * V_field[ele_idx, j, 1] * U_field[ele_idx, r, 1] * V_field[ele_idx, s, 1]
                        dPdF[ele_idx, ij, rs] = dPdF_tmp * dt * dt * vol0
    else:
        U, sig, V = svd(F)
        # U, sig, V = my_svd(F)
        for row_idx in ti.static(range(3)):
            for col_idx in ti.static(range(3)):
                U_field[ele_idx, row_idx, col_idx] = U[row_idx, col_idx]
                V_field[ele_idx, row_idx, col_idx] = V[row_idx, col_idx]

        sigma = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])
        sigmaProd = sigma[0] * sigma[1] * sigma[2]
        dE_div_dsigma = elasticity_gradient(sig, la, mu)
        d2E_div_dsigma2 = project_pd(elasticity_hessian(sig, la, mu))

        leftCoef = mu - la / 2 * sigma[2] * (sigmaProd - 1)
        rightCoef = dE_div_dsigma[0] + dE_div_dsigma[1]
        sum_sigma = ti.max(sigma[0] + sigma[1], 0.000001)
        rightCoef /= (2 * sum_sigma)
        B0 = project_pd(ti.Matrix([[leftCoef + rightCoef, leftCoef - rightCoef], [leftCoef - rightCoef, leftCoef + rightCoef]]))

        leftCoef = mu - la / 2 * sigma[0] * (sigmaProd - 1)
        rightCoef = dE_div_dsigma[1] + dE_div_dsigma[2]
        sum_sigma = ti.max(sigma[1] + sigma[2], 0.000001)
        rightCoef /= (2 * sum_sigma)
        B1 = project_pd(ti.Matrix([[leftCoef + rightCoef, leftCoef - rightCoef], [leftCoef - rightCoef, leftCoef + rightCoef]]))

        leftCoef = mu - la / 2 * sigma[1] * (sigmaProd - 1)
        rightCoef = dE_div_dsigma[2] + dE_div_dsigma[0]
        sum_sigma = ti.max(sigma[2] + sigma[0], 0.000001)
        rightCoef /= (2 * sum_sigma)
        B2 = project_pd(ti.Matrix([[leftCoef + rightCoef, leftCoef - rightCoef], [leftCoef - rightCoef, leftCoef + rightCoef]]))

        M_field[ele_idx, 0, 0] = d2E_div_dsigma2[0, 0]
        M_field[ele_idx, 0, 4] = d2E_div_dsigma2[0, 1]
        M_field[ele_idx, 0, 8] = d2E_div_dsigma2[0, 2]
        M_field[ele_idx, 4, 0] = d2E_div_dsigma2[1, 0]
        M_field[ele_idx, 4, 4] = d2E_div_dsigma2[1, 1]
        M_field[ele_idx, 4, 8] = d2E_div_dsigma2[1, 2]
        M_field[ele_idx, 8, 0] = d2E_div_dsigma2[2, 0]
        M_field[ele_idx, 8, 4] = d2E_div_dsigma2[2, 1]
        M_field[ele_idx, 8, 8] = d2E_div_dsigma2[2, 2]
        M_field[ele_idx, 1, 1] = B0[0, 0]
        M_field[ele_idx, 1, 3] = B0[0, 1]
        M_field[ele_idx, 3, 1] = B0[1, 0]
        M_field[ele_idx, 3, 3] = B0[1, 1]
        M_field[ele_idx, 5, 5] = B1[0, 0]
        M_field[ele_idx, 5, 7] = B1[0, 1]
        M_field[ele_idx, 7, 5] = B1[1, 0]
        M_field[ele_idx, 7, 7] = B1[1, 1]
        M_field[ele_idx, 2, 2] = B2[1, 1]
        M_field[ele_idx, 2, 6] = B2[1, 0]
        M_field[ele_idx, 6, 2] = B2[0, 1]
        M_field[ele_idx, 6, 6] = B2[0, 0]
        for j in range(3):
            for i in range(3):
                for s in range(3):
                    for r in range(3):
                        ij = j * 3 + i
                        rs = s * 3 + r
                        dPdF_tmp = M_field[ele_idx, 0, 0] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 0] * U_field[ele_idx, r, 0] * V_field[ele_idx, s, 0] + M_field[ele_idx, 0, 4] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 0] * \
                                       U_field[ele_idx, r, 1] * V_field[ele_idx, s, 1] + M_field[ele_idx, 0, 8] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 0] * U_field[ele_idx, r, 2] * V_field[ele_idx, s, 2] + M_field[ele_idx, 4, 0] * \
                                       U_field[ele_idx, i, 1] * V_field[ele_idx, j, 1] * U_field[ele_idx, r, 0] * V_field[ele_idx, s, 0] + M_field[ele_idx, 4, 4] * U_field[ele_idx, i, 1] * V_field[ele_idx, j, 1] * U_field[ele_idx, r, 1] * \
                                       V_field[ele_idx, s, 1] + M_field[ele_idx, 4, 8] * U_field[ele_idx, i, 1] * V_field[ele_idx, j, 1] * U_field[ele_idx, r, 2] * V_field[ele_idx, s, 2] + M_field[ele_idx, 8, 0] * U_field[ele_idx, i, 2] * \
                                       V_field[ele_idx, j, 2] * U_field[ele_idx, r, 0] * V_field[ele_idx, s, 0] + M_field[ele_idx, 8, 4] * U_field[ele_idx, i, 2] * V_field[ele_idx, j, 2] * U_field[ele_idx, r, 1] * V_field[ele_idx, s, 1] + \
                                       M_field[ele_idx, 8, 8] * U_field[ele_idx, i, 2] * V_field[ele_idx, j, 2] * U_field[ele_idx, r, 2] * V_field[ele_idx, s, 2] + M_field[ele_idx, 1, 1] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 1] * \
                                       U_field[ele_idx, r, 0] * V_field[ele_idx, s, 1] + M_field[ele_idx, 1, 3] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 1] * U_field[ele_idx, r, 1] * V_field[ele_idx, s, 0] + M_field[ele_idx, 3, 1] * \
                                       U_field[ele_idx, i, 1] * V_field[ele_idx, j, 0] * U_field[ele_idx, r, 0] * V_field[ele_idx, s, 1] + M_field[ele_idx, 3, 3] * U_field[ele_idx, i, 1] * V_field[ele_idx, j, 0] * U_field[ele_idx, r, 1] * \
                                       V_field[ele_idx, s, 0] + M_field[ele_idx, 5, 5] * U_field[ele_idx, i, 1] * V_field[ele_idx, j, 2] * U_field[ele_idx, r, 1] * V_field[ele_idx, s, 2] + M_field[ele_idx, 5, 7] * U_field[ele_idx, i, 1] * \
                                       V_field[ele_idx, j, 2] * U_field[ele_idx, r, 2] * V_field[ele_idx, s, 1] + M_field[ele_idx, 7, 5] * U_field[ele_idx, i, 2] * V_field[ele_idx, j, 1] * U_field[ele_idx, r, 1] * V_field[ele_idx, s, 2] + \
                                       M_field[ele_idx, 7, 7] * U_field[ele_idx, i, 2] * V_field[ele_idx, j, 1] * U_field[ele_idx, r, 2] * V_field[ele_idx, s, 1] + M_field[ele_idx, 2, 2] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 2] * \
                                       U_field[ele_idx, r, 0] * V_field[ele_idx, s, 2] + M_field[ele_idx, 2, 6] * U_field[ele_idx, i, 0] * V_field[ele_idx, j, 2] * U_field[ele_idx, r, 2] * V_field[ele_idx, s, 0] + M_field[ele_idx, 6, 2] * \
                                       U_field[ele_idx, i, 2] * V_field[ele_idx, j, 0] * U_field[ele_idx, r, 0] * V_field[ele_idx, s, 2] + M_field[ele_idx, 6, 6] * U_field[ele_idx, i, 2] * V_field[ele_idx, j, 0] * U_field[ele_idx, r, 2] * \
                                       V_field[ele_idx, s, 0]
                        dPdF[ele_idx, ij, rs] = dPdF_tmp * dt * dt * vol0