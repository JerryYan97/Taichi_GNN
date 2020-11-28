import sys, os, time
sys.path.insert(0, "../../build")
from JGSL_WATER import *
import taichi as ti
import numpy as np
import pymesh
from pdfix import*

mesh, dirichlet, mesh_scale, mesh_offset = read(1)

ti.init(arch=ti.gpu, default_fp=ti.f64)



