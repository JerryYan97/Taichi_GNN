import taichi as ti
import os
from .external_func import *

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


@ti.func
def cofactor(F):
    if ti.static(F.n == 2):
        return ti.Matrix([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]])
    else:
        return ti.Matrix([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]],
                          [F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1]],
                          [F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])


@ti.func
def svd(F):
    if ti.static(F.n == 2):
        F00, F01, F10, F11 = F[0, 0], F[0, 1], F[1, 0], F[1, 1]
        U00, U01, U10, U11 = 0.0, 0.0, 0.0, 0.0
        s00, s01, s10, s11 = 0.0, 0.0, 0.0, 0.0
        V00, V01, V10, V11 = 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.svd_2,
                              args=(F00, F01, F10, F11),
                              outputs=(U00, U01, U10, U11, s00, s01, s10, s11, V00, V01, V10, V11))
        return ti.Matrix([[U00, U01], [U10, U11]]), ti.Matrix([[s00, s01], [s10, s11]]), ti.Matrix([[V00, V01], [V10, V11]])
    if ti.static(F.n == 3):
        F00, F01, F02, F10, F11, F12, F20, F21, F22 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
        U00, U01, U02, U10, U11, U12, U20, U21, U22 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        s00, s01, s02, s10, s11, s12, s20, s21, s22 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        V00, V01, V02, V10, V11, V12, V20, V21, V22 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.svd_3,
                              args=(F00, F01, F02, F10, F11, F12, F20, F21, F22),
                              outputs=(U00, U01, U02, U10, U11, U12, U20, U21, U22, s00, s01, s02, s10, s11, s12, s20, s21, s22, V00, V01, V02, V10, V11, V12, V20, V21, V22))
        return ti.Matrix([[U00, U01, U02], [U10, U11, U12], [U20, U21, U22]]), ti.Matrix([[s00, s01, s02], [s10, s11, s12], [s20, s21, s22]]), ti.Matrix([[V00, V01, V02], [V10, V11, V12], [V20, V21, V22]])



@ti.func
def project_pd(F):
    if ti.static(F.n == 2):
        return make_pd(F)
    if ti.static(F.n == 3):
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
        out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.project_pd_3, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8))
        return ti.Matrix([[out_0, out_1, out_2], [out_3, out_4, out_5], [out_6, out_7, out_8]])
    if ti.static(F.n == 4):
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[3, 0], F[3, 1], F[3, 2], F[3, 3]
        out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.project_pd_4, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15))
        return ti.Matrix([[out_0, out_1, out_2, out_3], [out_4, out_5, out_6, out_7], [out_8, out_9, out_10, out_11], [out_12, out_13, out_14, out_15]])
    if ti.static(F.n == 6):
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5]
        out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.project_pd_6, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35))
        return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5], [out_6, out_7, out_8, out_9, out_10, out_11], [out_12, out_13, out_14, out_15, out_16, out_17], [out_18, out_19, out_20, out_21, out_22, out_23], [out_24, out_25, out_26, out_27, out_28, out_29], [out_30, out_31, out_32, out_33, out_34, out_35]])
    if ti.static(F.n == 9):
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[0, 6], F[0, 7], F[0, 8], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[1, 6], F[1, 7], F[1, 8], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[2, 6], F[2, 7], F[2, 8], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[3, 6], F[3, 7], F[3, 8], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[4, 6], F[4, 7], F[4, 8], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], F[5, 6], F[5, 7], F[5, 8], F[6, 0], F[6, 1], F[6, 2], F[6, 3], F[6, 4], F[6, 5], F[6, 6], F[6, 7], F[6, 8], F[7, 0], F[7, 1], F[7, 2], F[7, 3], F[7, 4], F[7, 5], F[7, 6], F[7, 7], F[7, 8], F[8, 0], F[8, 1], F[8, 2], F[8, 3], F[8, 4], F[8, 5], F[8, 6], F[8, 7], F[8, 8]
        out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.project_pd_9, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80))
        return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8], [out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17], [out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26], [out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35], [out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44], [out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53], [out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62], [out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71], [out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80]])
    if ti.static(F.n == 12):
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84, in_85, in_86, in_87, in_88, in_89, in_90, in_91, in_92, in_93, in_94, in_95, in_96, in_97, in_98, in_99, in_100, in_101, in_102, in_103, in_104, in_105, in_106, in_107, in_108, in_109, in_110, in_111, in_112, in_113, in_114, in_115, in_116, in_117, in_118, in_119, in_120, in_121, in_122, in_123, in_124, in_125, in_126, in_127, in_128, in_129, in_130, in_131, in_132, in_133, in_134, in_135, in_136, in_137, in_138, in_139, in_140, in_141, in_142, in_143 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[0, 6], F[0, 7], F[0, 8], F[0, 9], F[0, 10], F[0, 11], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[1, 6], F[1, 7], F[1, 8], F[1, 9], F[1, 10], F[1, 11], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[2, 6], F[2, 7], F[2, 8], F[2, 9], F[2, 10], F[2, 11], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[3, 6], F[3, 7], F[3, 8], F[3, 9], F[3, 10], F[3, 11], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[4, 6], F[4, 7], F[4, 8], F[4, 9], F[4, 10], F[4, 11], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], F[5, 6], F[5, 7], F[5, 8], F[5, 9], F[5, 10], F[5, 11], F[6, 0], F[6, 1], F[6, 2], F[6, 3], F[6, 4], F[6, 5], F[6, 6], F[6, 7], F[6, 8], F[6, 9], F[6, 10], F[6, 11], F[7, 0], F[7, 1], F[7, 2], F[7, 3], F[7, 4], F[7, 5], F[7, 6], F[7, 7], F[7, 8], F[7, 9], F[7, 10], F[7, 11], F[8, 0], F[8, 1], F[8, 2], F[8, 3], F[8, 4], F[8, 5], F[8, 6], F[8, 7], F[8, 8], F[8, 9], F[8, 10], F[8, 11], F[9, 0], F[9, 1], F[9, 2], F[9, 3], F[9, 4], F[9, 5], F[9, 6], F[9, 7], F[9, 8], F[9, 9], F[9, 10], F[9, 11], F[10, 0], F[10, 1], F[10, 2], F[10, 3], F[10, 4], F[10, 5], F[10, 6], F[10, 7], F[10, 8], F[10, 9], F[10, 10], F[10, 11], F[11, 0], F[11, 1], F[11, 2], F[11, 3], F[11, 4], F[11, 5], F[11, 6], F[11, 7], F[11, 8], F[11, 9], F[11, 10], F[11, 11]
        out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80, out_81, out_82, out_83, out_84, out_85, out_86, out_87, out_88, out_89, out_90, out_91, out_92, out_93, out_94, out_95, out_96, out_97, out_98, out_99, out_100, out_101, out_102, out_103, out_104, out_105, out_106, out_107, out_108, out_109, out_110, out_111, out_112, out_113, out_114, out_115, out_116, out_117, out_118, out_119, out_120, out_121, out_122, out_123, out_124, out_125, out_126, out_127, out_128, out_129, out_130, out_131, out_132, out_133, out_134, out_135, out_136, out_137, out_138, out_139, out_140, out_141, out_142, out_143 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.project_pd_12, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84, in_85, in_86, in_87, in_88, in_89, in_90, in_91, in_92, in_93, in_94, in_95, in_96, in_97, in_98, in_99, in_100, in_101, in_102, in_103, in_104, in_105, in_106, in_107, in_108, in_109, in_110, in_111, in_112, in_113, in_114, in_115, in_116, in_117, in_118, in_119, in_120, in_121, in_122, in_123, in_124, in_125, in_126, in_127, in_128, in_129, in_130, in_131, in_132, in_133, in_134, in_135, in_136, in_137, in_138, in_139, in_140, in_141, in_142, in_143), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23, out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35, out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47, out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59, out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71, out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80, out_81, out_82, out_83, out_84, out_85, out_86, out_87, out_88, out_89, out_90, out_91, out_92, out_93, out_94, out_95, out_96, out_97, out_98, out_99, out_100, out_101, out_102, out_103, out_104, out_105, out_106, out_107, out_108, out_109, out_110, out_111, out_112, out_113, out_114, out_115, out_116, out_117, out_118, out_119, out_120, out_121, out_122, out_123, out_124, out_125, out_126, out_127, out_128, out_129, out_130, out_131, out_132, out_133, out_134, out_135, out_136, out_137, out_138, out_139, out_140, out_141, out_142, out_143))
        return ti.Matrix([[out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11], [out_12, out_13, out_14, out_15, out_16, out_17, out_18, out_19, out_20, out_21, out_22, out_23], [out_24, out_25, out_26, out_27, out_28, out_29, out_30, out_31, out_32, out_33, out_34, out_35], [out_36, out_37, out_38, out_39, out_40, out_41, out_42, out_43, out_44, out_45, out_46, out_47], [out_48, out_49, out_50, out_51, out_52, out_53, out_54, out_55, out_56, out_57, out_58, out_59], [out_60, out_61, out_62, out_63, out_64, out_65, out_66, out_67, out_68, out_69, out_70, out_71], [out_72, out_73, out_74, out_75, out_76, out_77, out_78, out_79, out_80, out_81, out_82, out_83], [out_84, out_85, out_86, out_87, out_88, out_89, out_90, out_91, out_92, out_93, out_94, out_95], [out_96, out_97, out_98, out_99, out_100, out_101, out_102, out_103, out_104, out_105, out_106, out_107], [out_108, out_109, out_110, out_111, out_112, out_113, out_114, out_115, out_116, out_117, out_118, out_119], [out_120, out_121, out_122, out_123, out_124, out_125, out_126, out_127, out_128, out_129, out_130, out_131], [out_132, out_133, out_134, out_135, out_136, out_137, out_138, out_139, out_140, out_141, out_142, out_143]])


@ti.func
def solve(F, rhs):
    if ti.static(F.n == 2):
        in_0, in_1, in_2, in_3, in_4, in_5 = F[0, 0], F[0, 1], F[1, 0], F[1, 1], rhs[0], rhs[1]
        out_0, out_1 = 0.0, 0.0
        ti.external_func_call(func=so.solve_2, args=(in_0, in_1, in_2, in_3, in_4, in_5), outputs=(out_0, out_1))
        return ti.Vector([out_0, out_1])
    if ti.static(F.n == 3):
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11 = F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2], rhs[0], rhs[1], rhs[2]
        out_0, out_1, out_2 = 0.0, 0.0, 0.0
        ti.external_func_call(func=so.solve_3, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11), outputs=(out_0, out_1, out_2))
        return ti.Vector([out_0, out_1, out_2])
    if ti.static(F.n == 4):
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[3, 0], F[3, 1], F[3, 2], F[3, 3], rhs[0], rhs[1], rhs[2], rhs[3]
        out_0, out_1, out_2, out_3 = 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.solve_4, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19), outputs=(out_0, out_1, out_2, out_3))
        return ti.Vector([out_0, out_1, out_2, out_3])
    if ti.static(F.n == 6):
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], rhs[0], rhs[1], rhs[2], rhs[3], rhs[4], rhs[5]
        out_0, out_1, out_2, out_3, out_4, out_5 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.solve_6, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41), outputs=(out_0, out_1, out_2, out_3, out_4, out_5))
        return ti.Vector([out_0, out_1, out_2, out_3, out_4, out_5])
    if ti.static(F.n == 9):
        in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84, in_85, in_86, in_87, in_88, in_89 = F[0, 0], F[0, 1], F[0, 2], F[0, 3], F[0, 4], F[0, 5], F[0, 6], F[0, 7], F[0, 8], F[1, 0], F[1, 1], F[1, 2], F[1, 3], F[1, 4], F[1, 5], F[1, 6], F[1, 7], F[1, 8], F[2, 0], F[2, 1], F[2, 2], F[2, 3], F[2, 4], F[2, 5], F[2, 6], F[2, 7], F[2, 8], F[3, 0], F[3, 1], F[3, 2], F[3, 3], F[3, 4], F[3, 5], F[3, 6], F[3, 7], F[3, 8], F[4, 0], F[4, 1], F[4, 2], F[4, 3], F[4, 4], F[4, 5], F[4, 6], F[4, 7], F[4, 8], F[5, 0], F[5, 1], F[5, 2], F[5, 3], F[5, 4], F[5, 5], F[5, 6], F[5, 7], F[5, 8], F[6, 0], F[6, 1], F[6, 2], F[6, 3], F[6, 4], F[6, 5], F[6, 6], F[6, 7], F[6, 8], F[7, 0], F[7, 1], F[7, 2], F[7, 3], F[7, 4], F[7, 5], F[7, 6], F[7, 7], F[7, 8], F[8, 0], F[8, 1], F[8, 2], F[8, 3], F[8, 4], F[8, 5], F[8, 6], F[8, 7], F[8, 8], rhs[0], rhs[1], rhs[2], rhs[3], rhs[4], rhs[5], rhs[6], rhs[7], rhs[8]
        out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ti.external_func_call(func=so.solve_9, args=(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38, in_39, in_40, in_41, in_42, in_43, in_44, in_45, in_46, in_47, in_48, in_49, in_50, in_51, in_52, in_53, in_54, in_55, in_56, in_57, in_58, in_59, in_60, in_61, in_62, in_63, in_64, in_65, in_66, in_67, in_68, in_69, in_70, in_71, in_72, in_73, in_74, in_75, in_76, in_77, in_78, in_79, in_80, in_81, in_82, in_83, in_84, in_85, in_86, in_87, in_88, in_89), outputs=(out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8))
        return ti.Vector([out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8])


# Taichi SVD and Ctypes SVD replacement.
# A transformation from wrapper.cpp and relevant files.
# Taichi large scale development problems:
# 1.

@ti.func
def Get_Eigen_Values(A_Sym, lambda_vec):
    return A_Sym


@ti.func
def my_polar_decomposition2d(A, gu_rowi, gu_rowk, gu_c, gu_s):
    x = ti.Vector([A[0, 0] + A[1, 1], A[1, 0] - A[0, 1]])
    S_Sym = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
    denominator = x.norm()
    gu_c = 1.0
    gu_s = 0.0
    if denominator != 0.0:
        # No need to use a tolerance here because x(0) and x(1) always have
        # smaller magnitude then denominator, therefore overflow never happens.
        gu_c = x[0] / denominator
        gu_s = -x[1] / denominator
    # auto& S = const_cast<MATRIX<T, 2>&>(S_Sym);
    # S = A;
    # R.rowRotation(S);
    for i in ti.static(range(2)):
        for j in ti.static(range(2)):
            S_Sym[i, j] = A[i, j]
    for j in ti.static(range(2)):
        # tau1 = A[ti.static(gu_rowi), j]
        # tau2 = A[ti.static(gu_rowk), j]
        tau1 = A[0, j]
        tau2 = A[1, j]
        # S_Sym[gu_rowi, j] = gu_c * tau1 - gu_s * tau2
        # S_Sym[gu_rowk, j] = gu_s * tau1 + gu_c * tau2
        S_Sym[0, j] = gu_c * tau1 - gu_s * tau2
        S_Sym[1, j] = gu_s * tau1 + gu_c * tau2
    return S_Sym, gu_c, gu_s


@ti.func
def my_svd2d(F):
    # ti.svd()
    # GIVENS_ROTATION < T > gv(0, 1);
    gv_rowi, gv_rowk, gv_c, gv_s = 0, 1, 1.0, 0.0
    # GIVENS_ROTATION < T > gu(0, 1);
    gu_rowi, gu_rowk, gu_c, gu_s = 0, 1, 1.0, 0.0

    # Singular_Value_Decomposition(A, gu, Sigma, gv);
    Sigma_vec = ti.Vector([0.0, 0.0])
    S_Sym, gu_c, gu_s = my_polar_decomposition2d(F, gu_rowi, gu_rowk, gu_c, gu_s)

    cosine, sine = 0.0, 0.0
    x, y, z = S_Sym[0, 0], S_Sym[0, 1], S_Sym[1, 1]
    y2 = y * y
    if y2 == 0:
        # S is already diagonal
        cosine = 1.0
        sine = 0.0
        Sigma_vec[0] = x
        Sigma_vec[1] = z
    else:
        tau = 0.5 * (x - z)
        w = ti.sqrt(tau * tau + y2)
        # w > y > 0
        t = 0.0
        if tau > 0.0:
            # tau + w > w > y > 0 ==> division is safe
            t = y / (tau + w)
        else:
            # tau - w < -w < -y < 0 ==> division is safe
            t = y / (tau - w)
        cosine = 1.0 / ti.sqrt(t * t + 1.0)
        sine = -t * cosine
        # V = [cosine -sine; sine cosine]
        # Sigma = V'SV. Only compute the diagonals for efficiency.
        # Also utilize symmetry of S and don't form V yet.
        c2 = cosine * cosine
        csy = 2.0 * cosine * sine * y
        s2 = sine * sine
        Sigma_vec[0] = c2 * x - csy + s2 * z
        Sigma_vec[1] = s2 * x + csy + c2 * z
    # Sorting
    # Polar already guarantees negative sign is on the small magnitude singular value.
    if Sigma_vec[0] < Sigma_vec[1]:
        tmp = Sigma_vec[0]
        Sigma_vec[0] = Sigma_vec[1]
        Sigma_vec[1] = tmp
        gv_c, gv_s = -sine, cosine
    else:
        gv_c, gv_s = cosine, sine
    # U *= V;
    new_c = gu_c * gv_c - gu_s * gv_s
    new_s = gu_s * gv_c + gu_c * gv_s
    gu_c = new_c
    gu_s = new_s

    # gu.fill(U);
    U = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
    U[0, 0] = gu_c
    U[1, 0] = -gu_s
    U[0, 1] = gu_s
    U[1, 1] = gu_c
    # gv.fill(V);
    V = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
    V[0, 0] = gv_c
    V[1, 0] = -gv_s
    V[0, 1] = gv_s
    V[1, 1] = gv_c

    return U, ti.Matrix([[Sigma_vec[0], 0.0], [0.0, Sigma_vec[1]]]), V


@ti.func
def Get_Eigen_Values(A_Sym):
    lambda_vec = ti.Vector([0.0, 0.0, 0.0])
    m = (1.0 / 3.0) * (A_Sym[0, 0] + A_Sym[1, 1] + A_Sym[2, 2])
    a00 = A_Sym[0, 0] - m
    a11 = A_Sym[1, 1] - m
    a22 = A_Sym[2, 2] - m
    a12_sqr = A_Sym[0, 1] * A_Sym[0, 1]
    a13_sqr = A_Sym[0, 2] * A_Sym[0, 2]
    a23_sqr = A_Sym[1, 2] * A_Sym[1, 2]
    p = (1.0 / 6.0) * (a00 * a00 + a11 * a11 + a22 * a22 + 2 * (a12_sqr + a13_sqr + a23_sqr))
    q = 0.5 * (a00 * (a11 * a22 - a23_sqr) - a11 * a13_sqr - a22 * a12_sqr) + A_Sym[0, 1] * A_Sym[0, 2] * A_Sym[1, 2]
    sqrt_p = ti.sqrt(p)
    disc = p ** 3 - q ** 2
    phi = (1.0 / 3.0) * ti.atan2(ti.sqrt(ti.max(0.0, disc)), q)
    c = ti.cos(phi)
    s = ti.sin(phi)
    sqrt_p_cos = sqrt_p * c
    root_three_sqrt_p_sin = ti.sqrt(3.0) * sqrt_p * s

    lambda_vec[0] = m + 2 * sqrt_p_cos
    lambda_vec[1] = m - sqrt_p_cos - root_three_sqrt_p_sin
    lambda_vec[2] = m - sqrt_p_cos + root_three_sqrt_p_sin

    if lambda_vec[0] < lambda_vec[1]:
        tmp = lambda_vec[0]
        lambda_vec[0] = lambda_vec[1]
        lambda_vec[1] = tmp
    if lambda_vec[1] < lambda_vec[2]:
        tmp = lambda_vec[1]
        lambda_vec[1] = lambda_vec[2]
        lambda_vec[2] = tmp
    if lambda_vec[0] < lambda_vec[1]:
        tmp = lambda_vec[0]
        lambda_vec[0] = lambda_vec[1]
        lambda_vec[1] = tmp

    return lambda_vec


# @ti.func
# def cofactor(A):
#     m = ti.Matrix([[0.0, 0.0, 0.0],
#                    [0.0, 0.0, 0.0],
#                    [0.0, 0.0, 0.0]])
#     m[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
#     m[0, 1] = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
#     m[0, 2] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
#     m[1, 0] = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
#     m[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
#     m[1, 2] = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]
#     m[2, 0] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
#     m[2, 1] = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]
#     m[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
#     return m


# inline static T Get_Max_Coeff(VECTOR<T, dim>& V, int& index)
@ti.func
def Get_Max_Coeff(V):
    max = V[0]
    index = 0
    for i in ti.static(range(V.n)):
        if max < V[i]:
            max = V[i]
            index = i
    return max, index


@ti.func
def Random_Access_3dVector(vec3d, idx):
    res_ele = 0.0
    if idx == 0:
        res_ele = vec3d[0]
    elif idx == 1:
        res_ele = vec3d[1]
    elif idx == 2:
        res_ele = vec3d[2]
    else:
        raise Exception("Random_Access_3dVector Access Error. Error idx:", idx)
    return res_ele


@ti.func
def Random_Access_3dMat_Col(mat3d, col_idx):
    res_vec = ti.Vector([0.0, 0.0, 0.0])
    if col_idx == 0:
        res_vec = ti.Vector([mat3d[0, 0], mat3d[1, 0], mat3d[2, 0]])
    elif col_idx == 1:
        res_vec = ti.Vector([mat3d[0, 1], mat3d[1, 1], mat3d[2, 1]])
    elif col_idx == 2:
        res_vec = ti.Vector([mat3d[0, 2], mat3d[1, 2], mat3d[2, 2]])
    else:
        # ti.static_print(col_idx)
        print("Random_Access_3dMat_Col Access Error. Error col_idx:", col_idx)
       #  raise Exception("Random_Access_3dMat_Col Access Error. Error col_idx:", col_idx)
    return res_vec


@ti.func
def Random_Access_3dMat_element(mat3d, row_idx, col_idx):
    res_ele = 0.0
    if row_idx == 0:
        if col_idx == 0:
            res_ele = mat3d[0, 0]
        elif col_idx == 1:
            res_ele = mat3d[0, 1]
        elif col_idx == 2:
            res_ele = mat3d[0, 2]
        else:
            raise Exception("Random_Access_3dMat_element Access Error. Error col_idx:", col_idx)
    elif row_idx == 1:
        if col_idx == 0:
            res_ele = mat3d[1, 0]
        elif col_idx == 1:
            res_ele = mat3d[1, 1]
        elif col_idx == 2:
            res_ele = mat3d[1, 2]
        else:
            raise Exception("Random_Access_3dMat_element Access Error. Error col_idx:", col_idx)
    elif row_idx == 2:
        if col_idx == 0:
            res_ele = mat3d[2, 0]
        elif col_idx == 1:
            res_ele = mat3d[2, 1]
        elif col_idx == 2:
            res_ele = mat3d[2, 2]
        else:
            raise Exception("Random_Access_3dMat_element Access Error. Error col_idx:", col_idx)
    else:
        raise Exception("Random_Access_3dMat_element Access Error. Error col_idx:", col_idx)
    return res_ele


@ti.func
def Random_Access_2dMat_element(mat2d, row_idx, col_idx):
    res_ele = 0.0
    if row_idx == 0:
        if col_idx == 0:
            res_ele = mat2d[0, 0]
        elif col_idx == 1:
            res_ele = mat2d[0, 1]
        else:
            print("Random_Access_3dMat_element Access Error. Error col_idx:", col_idx)
    elif row_idx == 1:
        if col_idx == 0:
            res_ele = mat2d[1, 0]
        elif col_idx == 1:
            res_ele = mat2d[1, 1]
        else:
            print("Random_Access_3dMat_element Access Error. Error col_idx:", col_idx)
    return res_ele


@ti.func
def Get_Eigen_Vectors(A_Sym, lambda_vec):
    flipped = False
    lambda_flip = ti.Vector([lambda_vec[0], lambda_vec[1], lambda_vec[2]])
    if (lambda_vec[0] - lambda_vec[1]) < (lambda_vec[1] - lambda_vec[2]):
        tmp = lambda_flip[0]
        lambda_flip[0] = lambda_flip[2]
        lambda_flip[2] = tmp
        flipped = True

    # get first eigenvector
    C1 = cofactor(A_Sym - ti.Matrix([[lambda_flip[0], 0.0, 0.0],
                                     [0.0, lambda_flip[0], 0.0],
                                     [0.0, 0.0, lambda_flip[0]]]))
    SquaredNorm = ti.Vector([0.0, 0.0, 0.0])
    for i_loop in ti.static(range(3)):
        for j_loop in ti.static(range(3)):
            SquaredNorm[i_loop] += C1[i_loop, j_loop] * C1[i_loop, j_loop]

    # int i = -1;
    # T norm2 = MATH_TOOLS::Get_Max_Coeff<T>(SquaredNorm, i);
    norm2, i = Get_Max_Coeff(SquaredNorm)
    v1 = ti.Vector([0.0, 0.0, 0.0])
    if norm2 != 0.0:
        one_over_sqrt = 1.0 / ti.sqrt(norm2)
        v1 = Random_Access_3dMat_Col(C1, i) * one_over_sqrt
    else:
        v1 = ti.Vector([1.0, 0.0, 0.0])

    # Form basis for orthogonal complement to v1, and reduce A to this space need this function.
    v1_orthogonal = Get_Unit_Orthogonal(v1)
    other_v = ti.Matrix.cols([v1_orthogonal, v1.cross(v1_orthogonal)])
    A_reduced = other_v.transpose() @ A_Sym @ other_v
    C3 = cofactor(A_reduced - ti.Matrix([[lambda_flip[2], 0.0],
                                         [0.0, lambda_flip[2]]]))
    SquaredNorm2 = ti.Vector([0.0, 0.0])
    for i_loop in ti.static(range(2)):
        for j_loop in ti.static(range(2)):
            SquaredNorm2[i_loop] += C3[i_loop, j_loop] * C3[i_loop, j_loop]
    norm2, j = Get_Max_Coeff(SquaredNorm2)

    v3 = ti.Vector([0.0, 0.0, 0.0])
    if norm2 != 0.0:
        one_over_sqrt = 1.0 / ti.sqrt(norm2)
        tmp = ti.Vector([0.0, 0.0, 0.0])
        C3_j_0 = Random_Access_2dMat_element(C3, j, 0)
        C3_j_1 = Random_Access_2dMat_element(C3, j, 1)
        tmp[0] = other_v[0, 0] * C3_j_0 + other_v[0, 1] * C3_j_1
        tmp[1] = other_v[1, 0] * C3_j_0 + other_v[1, 1] * C3_j_1
        tmp[2] = other_v[2, 0] * C3_j_0 + other_v[2, 1] * C3_j_1
        v3 = tmp * one_over_sqrt
    else:
        v3 = ti.Vector([other_v[0, 0], other_v[1, 0], other_v[2, 0]])

    v2 = v3.cross(v1)

    V = ti.Matrix([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
    if flipped:
        V = ti.Matrix.cols([v3, v2, -1.0 * v1])
    else:
        V = ti.Matrix.cols([v1, v2, v3])

    return V


@ti.func
def Is_Much_Smaller_Than(a, b):
    res = False
    if ti.abs(a) < ti.abs(b) * 1e-12:
        res = True
    else:
        res = False
    return res


@ti.func
def Get_Unit_Orthogonal(V):
    res_vec = ti.Vector([0.0, 0.0, 0.0])
    if not Is_Much_Smaller_Than(V[0], V[2]) or not Is_Much_Smaller_Than(V[1], V[2]):
        invnm = 1.0 / ti.sqrt(V[0] * V[0] + V[1] * V[1])
        res_vec = ti.Vector([-V[1] * invnm, V[0] * invnm, 0.0])
    else:
        invnm = 1.0 / ti.sqrt(V[1] * V[1] + V[2] * V[2])
        res_vec = ti.Vector([0.0, -V[2] * invnm, V[1] * invnm])
    return res_vec


@ti.func
def my_svd3d(A):
    # sd = ti.Vector([0.0, 0.0, 0.0])
    # Ud = ti.Matrix([[0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0],
    #                 [0.0, 0.0, 0.0]])
    Vd = ti.Matrix([[0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]])
    Sigma_vec = ti.Vector([0.0, 0.0, 0.0])
    lambda_vec = ti.Vector([0.0, 0.0, 0.0])

    # MATRIX<T, 3> A_Sym = A.transpose() * A;
    A_Sym = A.transpose() @ A
    # Get_Eigen_Values(A_Sym, lambda);
    lambda_vec = Get_Eigen_Values(A_Sym)
    # Get_Eigen_Vectors(A_Sym, lambda, V);
    Vd = Get_Eigen_Vectors(A_Sym, lambda_vec)

    # compute singular values
    if lambda_vec[2] < 0.0:
        for i in ti.static(range(3)):
            if lambda_vec[i] >= 0.0:
                lambda_vec[i] = lambda_vec[i]
            else:
                lambda_vec[i] = 0.0
    Sigma_vec = ti.sqrt(lambda_vec)
    if A.determinant() < 0.0:
        Sigma_vec[2] = - Sigma_vec[2]

    # compute singular vectors
    U_col_0 = A @ ti.Vector([Vd[0, 0], Vd[1, 0], Vd[2, 0]])
    norm = U_col_0.norm()
    if norm != 0:
        one_over_norm = 1.0 / norm
        U_col_0 = U_col_0 * one_over_norm
    else:
        U_col_0 = ti.Vector([1.0, 0.0, 0.0])

    # VECTOR<double, 3> v1_orthogonal = MATH_TOOLS::Get_Unit_Orthogonal<T>(U(0));
    v1_orthogonal = Get_Unit_Orthogonal(U_col_0)
    other_v_col_0 = ti.Vector([v1_orthogonal[0], v1_orthogonal[1], v1_orthogonal[2]])
    other_v_col_1 = U_col_0.cross(v1_orthogonal)
    other_v = ti.Matrix.cols([other_v_col_0, other_v_col_1])
    w = other_v.transpose() @ A @ ti.Vector([Vd[0, 1], Vd[1, 1], Vd[2, 1]])
    norm = w.norm()
    if norm != 0:
        one_over_norm = 1.0 / norm
        w = w * one_over_norm
    else:
        w = ti.Vector([1.0, 0.0])

    U_col_1 = ti.Vector([other_v[0, 0] * w[0] + other_v[0, 1] * w[1],
                         other_v[1, 0] * w[0] + other_v[1, 1] * w[1],
                         other_v[2, 0] * w[0] + other_v[2, 1] * w[1]])
    U_col_2 = U_col_0.cross(U_col_1)
    Ud = ti.Matrix.cols([U_col_0, U_col_1, U_col_2])

    Sigma_mat = ti.Matrix([[Sigma_vec[0], 0.0, 0.0],
                           [0.0, Sigma_vec[1], 0.0],
                           [0.0, 0.0, Sigma_vec[2]]])
    return Ud, Sigma_mat, Vd


@ti.func
def my_svd(F):
    # ti.svd()
    if ti.static(F.n == 2):
        ret = my_svd2d(F)
        return ret
    elif ti.static(F.n == 3):
        return my_svd3d(F)
        # raise Exception("SVD 3D is not implemented.")
    else:
        raise Exception("SVD only supports 2D and 3D matrices.")