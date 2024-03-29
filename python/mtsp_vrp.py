from ctypes import *
import numpy as np
from numpy.ctypeslib import as_array, ndpointer
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
    c_int, # optimizationMode
    c_int, # timeout
    c_size_t, # numberOfThreads
    POINTER(c_double), # lowerBound
    POINTER(c_double), # upperBound
    ndpointer(c_size_t, flags='C_CONTIGUOUS'), # paths
    ndpointer(c_size_t, flags='C_CONTIGUOUS'), # pathOffsets
    c_void_p # fractionalCallback
]

error_code_map = {
    -1: 'Timeout, no result',
    -2: 'Infeasible',
    -3: 'Invalid input size',
    -4: 'Invalid input pointer',
    -5: 'Cyclic dependencies',
    -6: 'Incompatible dependencies'
}

def solve_mtsp_vrp(start_positions, end_positions, weights, optimization_mode, timeout, number_of_threads=0, fractional_callback=None):
    A = len(start_positions)
    N = len(weights)
    start_positions = np.array(start_positions, dtype=np.uint64)
    end_positions = np.array(end_positions, dtype=np.uint64)
    weights = np.array(weights, dtype=np.int32)
    optimization_mode = 1 if str(optimization_mode).upper() in ['MAX', '1'] else 0
    number_of_threads = int(number_of_threads)
    lb = c_double(0)
    ub = c_double(0)
    pathsBuffer = np.zeros(shape=(2 * A + N,), dtype=np.uint64)
    offsets = np.zeros(shape=(A,), dtype=np.uint64)

    if fractional_callback:
        @CFUNCTYPE(c_int, POINTER(c_double))
        def fractional_callback_c(fractional_values):
            x = as_array(fractional_values, shape=(A, N, N))
            fractional_callback(np.copy(x))
            return 0
    else:
        fractional_callback_c = None

    result = _solve_mtsp_vrp(A, N, start_positions, end_positions, weights, optimization_mode, timeout,
                             number_of_threads, byref(lb), byref(ub), pathsBuffer, offsets, fractional_callback_c)
    if result < 0:
        error = error_code_map.get(result, f'Unknown error code: {result}')
        raise Exception(error)

    paths = []
    lengths = []
    for a in range(A):
        start = offsets[a]
        end = offsets[a+1] if a+1 < A else N
        path = list(np.array(pathsBuffer[start:end]))
        length = float(np.sum(weights[path[:-1], path[1:]]))
        if len(path) >= 2 and start_positions[a] == end_positions[a]:
            length += weights[path[-1], path[0]]
        paths.append(path)
        lengths.append(length)

    return paths, lengths, lb.value, ub.value
