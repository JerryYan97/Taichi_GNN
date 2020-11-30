import taichi as ti
import os
import ctypes


@ti.func
def make_pd(symMtr):
    a = symMtr[0, 0]
    b = (symMtr[0, 1] + symMtr[1, 0]) / 2.0
    d = symMtr[1, 1]
    b2 = b * b
    D = a * d - b2
    T_div_2 = (a + d) / 2.0
    sqrtTT4D = ti.sqrt(T_div_2 * T_div_2 - D)
    L2 = T_div_2 - sqrtTT4D
    if L2 < 0.0:
        L1 = T_div_2 + sqrtTT4D
        if L1 <= 0.0:
            symMtr = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        else:
            if b2 == 0.0:
                symMtr = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                symMtr[0, 0] = L1
            else:
                L1md = L1 - d
                L1md_div_L1 = L1md / L1
                symMtr[0, 0] = L1md_div_L1 * L1md
                symMtr[0, 1] = b * L1md_div_L1
                symMtr[1, 0] = b * L1md_div_L1
                symMtr[1, 1] = b2 / L1
    return symMtr


# so = ctypes.CDLL("./a.so")
os.path.dirname(os.path.dirname(os.path.abspath(__file__)) + "a.so")

@ti.func
def singular_value_decomposition(F):
    F00, F01, F10, F11 = F(0, 0), F(0, 1), F(1, 0), F(1, 1)
    U00, U01, U10, U11 = 0.0, 0.0, 0.0, 0.0
    s00, s01, s10, s11 = 0.0, 0.0, 0.0, 0.0
    V00, V01, V10, V11 = 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.svd,
                          args=(F00, F01, F10, F11),
                          outputs=(U00, U01, U10, U11, s00, s01, s10, s11, V00, V01, V10, V11))
    return ti.Matrix([[U00, U01], [U10, U11]]), ti.Matrix([[s00, s01], [s10, s11]]), ti.Matrix([[V00, V01], [V10, V11]])


@ti.func
def project_pd(F, diagonal):
    F00, F01, F02, F03, F04, F05 = F(0, 0), F(0, 1), F(0, 2), F(0, 3), F(0, 4), F(0, 5)
    F10, F11, F12, F13, F14, F15 = F(1, 0), F(1, 1), F(1, 2), F(1, 3), F(1, 4), F(1, 5)
    F20, F21, F22, F23, F24, F25 = F(2, 0), F(2, 1), F(2, 2), F(2, 3), F(2, 4), F(2, 5)
    F30, F31, F32, F33, F34, F35 = F(3, 0), F(3, 1), F(3, 2), F(3, 3), F(3, 4), F(3, 5)
    F40, F41, F42, F43, F44, F45 = F(4, 0), F(4, 1), F(4, 2), F(4, 3), F(4, 4), F(4, 5)
    F50, F51, F52, F53, F54, F55 = F(5, 0), F(5, 1), F(5, 2), F(5, 3), F(5, 4), F(5, 5)
    PF00, PF01, PF02, PF03, PF04, PF05 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF10, PF11, PF12, PF13, PF14, PF15 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF20, PF21, PF22, PF23, PF24, PF25 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF30, PF31, PF32, PF33, PF34, PF35 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF40, PF41, PF42, PF43, PF44, PF45 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    PF50, PF51, PF52, PF53, PF54, PF55 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.project_pd,
                          args=(F00, F01, F02, F03, F04, F05,
                                F10, F11, F12, F13, F14, F15,
                                F20, F21, F22, F23, F24, F25,
                                F30, F31, F32, F33, F34, F35,
                                F40, F41, F42, F43, F44, F45,
                                F50, F51, F52, F53, F54, F55, diagonal),
                          outputs=(PF00, PF01, PF02, PF03, PF04, PF05,
                                   PF10, PF11, PF12, PF13, PF14, PF15,
                                   PF20, PF21, PF22, PF23, PF24, PF25,
                                   PF30, PF31, PF32, PF33, PF34, PF35,
                                   PF40, PF41, PF42, PF43, PF44, PF45,
                                   PF50, PF51, PF52, PF53, PF54, PF55))
    return ti.Matrix([[PF00, PF01, PF02, PF03, PF04, PF05],
                      [PF10, PF11, PF12, PF13, PF14, PF15],
                      [PF20, PF21, PF22, PF23, PF24, PF25],
                      [PF30, PF31, PF32, PF33, PF34, PF35],
                      [PF40, PF41, PF42, PF43, PF44, PF45],
                      [PF50, PF51, PF52, PF53, PF54, PF55]])



@ti.func
def project_pd64(F, diagonal):
    F00, F01, F02, F03, F04, F05 = F(0, 0), F(0, 1), F(0, 2), F(0, 3), F(0, 4), F(0, 5)
    F10, F11, F12, F13, F14, F15 = F(1, 0), F(1, 1), F(1, 2), F(1, 3), F(1, 4), F(1, 5)
    F20, F21, F22, F23, F24, F25 = F(2, 0), F(2, 1), F(2, 2), F(2, 3), F(2, 4), F(2, 5)
    F30, F31, F32, F33, F34, F35 = F(3, 0), F(3, 1), F(3, 2), F(3, 3), F(3, 4), F(3, 5)
    F40, F41, F42, F43, F44, F45 = F(4, 0), F(4, 1), F(4, 2), F(4, 3), F(4, 4), F(4, 5)
    F50, F51, F52, F53, F54, F55 = F(5, 0), F(5, 1), F(5, 2), F(5, 3), F(5, 4), F(5, 5)
    PF00, PF01, PF02, PF03 = 0.0, 0.0, 0.0, 0.0
    PF10, PF11, PF12, PF13 = 0.0, 0.0, 0.0, 0.0
    PF20, PF21, PF22, PF23 = 0.0, 0.0, 0.0, 0.0
    PF30, PF31, PF32, PF33 = 0.0, 0.0, 0.0, 0.0
    ti.external_func_call(func=so.project_pd64,
                          args=(F00, F01, F02, F03, F04, F05,
                                F10, F11, F12, F13, F14, F15,
                                F20, F21, F22, F23, F24, F25,
                                F30, F31, F32, F33, F34, F35,
                                F40, F41, F42, F43, F44, F45,
                                F50, F51, F52, F53, F54, F55, diagonal),
                          outputs=(PF00, PF01, PF02, PF03,
                                   PF10, PF11, PF12, PF13,
                                   PF20, PF21, PF22, PF23,
                                   PF30, PF31, PF32, PF33))
    return ti.Matrix([[PF00, PF01, PF02, PF03],
                      [PF10, PF11, PF12, PF13],
                      [PF20, PF21, PF22, PF23],
                      [PF30, PF31, PF32, PF33]])
