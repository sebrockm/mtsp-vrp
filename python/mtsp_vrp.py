from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer, as_array
from os import path

with open(path.join(path.dirname(path.abspath(__file__)), '_mtsp_vrp_c_lib_path.txt')) as f:
    _mtsp_vrp_c_lib_path = f.readline()

_solve_mtsp_vrp = cdll.LoadLibrary(_mtsp_vrp_c_lib_path).solve_mtsp_vrp
_solve_mtsp_vrp.restype = c_int
_solve_mtsp_vrp.argtypes = [
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
    ndpointer(c_size_t, flags='C_CONTIGUOUS'), # pathOffsets
    c_void_p # fractionalCallback
]

def solve_mtsp_vrp(start_positions, end_positions, weights, timeout, fractional_callback=None):
    A = len(start_positions)
    N = len(weights)
    start_positions = np.array(start_positions, dtype=np.uint64)
    end_positions = np.array(end_positions, dtype=np.uint64)
    weights = np.array(weights, dtype=np.int32)
    number_of_threads = 0
    lb = c_double(0)
    ub = c_double(0)
    pathsBuffer = np.zeros(shape=(N,), dtype=np.uint64)
    offsets = np.zeros(shape=(A,), dtype=np.uint64)

    if fractional_callback:
        @CFUNCTYPE(c_int, POINTER(c_double), c_size_t, c_size_t)
        def fractional_callback_c(fractional_values, A, N):
            fractional_callback(np.copy(as_array(fractional_values, shape=(A, N, N))))
            return 0
    else:
        fractional_callback = None

    result = _solve_mtsp_vrp(A, N, start_positions, end_positions, weights, timeout, number_of_threads,
                             byref(lb), byref(ub), pathsBuffer, offsets, fractional_callback_c)
    if result < 0:
        print(f'error: {result}')
        return None, None, result, result

    paths = []
    lengths = []
    for a in range(A):
        start = offsets[a]
        end = offsets[a+1] if a+1 < A else N
        path = np.array(pathsBuffer[start:end])
        length = np.sum(weights[path[:-1], path[1:]])
        if start_positions[a] == end_positions[a]:
            length += weights[path[-1], path[0]]
        paths.append(path)
        lengths.append(length)

    return paths, lengths, lb.value, ub.value
