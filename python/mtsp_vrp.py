from ctypes import *
from numpy.ctypeslib import ndpointer
from os import path

with open(path.join(path.dirname(path.abspath(__file__)), '_mtsp_vrp_c_lib_path.txt')) as f:
    mtsp_vrp_c_lib_path = f.readline()

solve_mtsp_vrp = cdll.LoadLibrary(mtsp_vrp_c_lib_path).solve_mtsp_vrp
solve_mtsp_vrp.restype = c_int
solve_mtsp_vrp.argtypes = [
    c_size_t, # numberOfAgents
    c_size_t, # numberOfNodes
    ndpointer(c_size_t, flags='C_CONTIGUOUS'), # start_positions
    ndpointer(c_size_t, flags='C_CONTIGUOUS'), # end_positions
    ndpointer(c_int, flags='C_CONTIGUOUS'), # weights
    c_int, # timeout
    c_size_t, # numberOfThreads
    POINTER(c_double), # lowerBound
    POINTER(c_double), # upperBound
    ndpointer(c_size_t, flags='C_CONTIGUOUS'), # paths
    ndpointer(c_size_t, flags='C_CONTIGUOUS') # pathOffsets
]
