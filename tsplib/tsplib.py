import tsplib95 as tsplib
import numpy as np
from ctypes import *
import os
import sys
import time

mtsp_vrp_dll = None

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()

        return ret, time2 - time1
    return wrap

@timing
def solve_mtsp(start_positions, end_positions, weights, timeout):
    A = len(start_positions)
    N = len(weights)
    pStarts = np.array(start_positions).astype(c_int).ctypes.data_as(POINTER(c_int))
    pEnds = np.array(end_positions).astype(c_int).ctypes.data_as(POINTER(c_int))
    W = np.array(weights).astype(c_double).ctypes.data_as(POINTER(c_double))
    lb = c_double(0)
    ub = c_double(0)
    pPaths = np.zeros(shape=(N,)).astype(c_int).ctypes.data_as(POINTER(c_int))
    pOffsets = np.zeros(shape=(A,)).astype(c_size_t).ctypes.data_as(POINTER(c_size_t))

    result = mtsp_vrp_dll.solve_mtsp_vrp(c_size_t(A), c_size_t(N), pStarts, pEnds, W, c_int(timeout), byref(lb), byref(ub), pPaths, pOffsets)
    if result < 0:
        return None, None, result

    paths = []
    lengths = []
    for a in range(A):
        start = pOffsets[a]
        end = pOffsets[a+1] if a+1 < A else N
        path = np.array(pPaths[start:end])
        length = np.sum(weights[path[:-1], path[1:]])
        paths.append(path)
        lengths.append(length)

    gap = ub.value / lb.value - 1
    return paths, lengths, gap

def main(dll_path, timeout_ms):
    global mtsp_vrp_dll
    mtsp_vrp_dll = cdll.LoadLibrary(dll_path)

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tsplib')
    files = [os.path.join(base, kind, f) for kind in ['sop', 'atsp', 'tsp'] for f in os.listdir(os.path.join(base, kind))]

    for f in files:
        base_name = os.path.basename(f)

        print(f'loading problem {base_name} ...')
        P = tsplib.load(f)
        N = P.dimension

        if N <= 200 and f[-4:] != '.sop': # increase once we are faster
            print('creating weight matrix...')
            if P.is_full_matrix():
                matrix = P.edge_weights[1:] if len(P.edge_weights[0]) == 1 else P.edge_weights
                weights = np.array(sum(matrix, []), dtype=int).reshape((N, N))
            else:
                weights = np.zeros((N, N), dtype=int)
                nodes = list(P.get_nodes())
                for i in range(N):
                    for j in range(N):
                        weights[i, j] = P.get_weight(nodes[i], nodes[j])

            print('looking for dependencies...')
            dependencies = sorted((i, j) for j, i in zip(*np.where(weights == -1)) if i != j)
            if (len(dependencies) > 0):
                print(f'ignoring {f} because it has dependencies')
                continue # dependencies are not supported yet

            print(f'starting solving {f} ...')
            (paths, lengths, gap), seconds = solve_mtsp(start_positions=[0], end_positions=[1], weights=weights, timeout=timeout_ms) # TODO: change to end_positions=[0] once supported
            if paths is None:
                print('solve_mtsp error:', gap)
                result_string = f'{base_name:<15s} N={N:>4d} A=1 mode=sum time=------s result=------- gap=-------\n'
            else:
                result_string = f'{base_name:<15s} N={N:>4d} A=1 mode=sum time={seconds:>6.3f}s result={lengths[0]:>7d} gap={gap:>7.2%}\n'
        else:
            result_string = f'{base_name:<15s} N={N:>4d} A=1 mode=sum time=------s result=------- gap=-------\n'
    
        print(result_string)
        with open(os.path.join(base, 'bench.txt'), 'a') as f:
            f.write(result_string)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
