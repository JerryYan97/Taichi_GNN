import ctypes
import os

so = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + "/a.so")
so_path = os.path.dirname(os.path.realpath(__file__)) + "/a.so"
print(so_path)
