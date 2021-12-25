import tsplib95 as tsplib
import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import *
import json
import os
import sys
import time

solve_mtsp_vrp = None

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
    start_positions = np.array(start_positions, dtype=np.int32)
    end_positions = np.array(end_positions, dtype=np.int32)
    weights = np.array(weights, dtype=np.int32)
    lb = c_double(0)
    ub = c_double(0)
    pathsBuffer = np.zeros(shape=(N,), dtype=np.int32)
    offsets = np.zeros(shape=(A,), dtype=np.uint64)

    print(f'solve_mtsp_vrp(A={A}, N={N}, start_positions, end_positions, weights, timeout={timeout}, byref(lb), byref(ub), pathsBuffer, offsets)')
    result = solve_mtsp_vrp(A, N, start_positions, end_positions, weights, timeout, byref(lb), byref(ub), pathsBuffer, offsets)
    if result < 0:
        return None, None, result, result

    print(pathsBuffer)
    print(offsets)

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

def main(dll_path, timeout_ms):
    global solve_mtsp_vrp
    solve_mtsp_vrp = cdll.LoadLibrary(dll_path).solve_mtsp_vrp
    solve_mtsp_vrp.restype = c_int
    solve_mtsp_vrp.argtypes = [
        c_size_t, # numberOfAgents
        c_size_t, # numberOfNodes
        c_size_t, # numberOfDependencies
        ndpointer(c_int, flags='C_CONTIGUOUS'), # start_positions
        ndpointer(c_int, flags='C_CONTIGUOUS'), # end_positions
        ndpointer(c_int, flags='C_CONTIGUOUS'), # weights
        c_int, # timeout
        POINTER(c_double), # lowerBound
        POINTER(c_double), # upperBound
        ndpointer(c_int, flags='C_CONTIGUOUS'), # paths
        ndpointer(c_size_t, flags='C_CONTIGUOUS') # pathOffsets
    ]
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tsplib')
    files = [os.path.join(base, kind, f) for kind in ['sop', 'atsp', 'tsp'] for f in os.listdir(os.path.join(base, kind))]

    with open(os.path.join(base, 'best-known-solutions.json')) as f:
        solutions = json.load(f)

    bench_file = os.path.join(base, 'bench.txt')
    try:
        os.remove(bench_file)
    except OSError:
        pass

    missing_best_known_solutions = ['ESC11.sop']

    for f in files:
        base_name = os.path.basename(f)
        problem_name, ext = os.path.splitext(base_name)
        kind = ext[1:]
        if kind not in ['tsp', 'atsp', 'sop']:
            continue
        if base_name in missing_best_known_solutions:
            best_lb = 0
            best_ub = float('inf')
        else:
            solution = solutions[kind][problem_name]
            if isinstance(solution, int):
                best_lb = solution
                best_ub = solution
            else:
                best_lb, best_ub = solution

        print(f'loading problem {base_name} ...')
        P = tsplib.load(f)
        N = P.dimension

        if N <= 1000: # increase once we are faster
            print('loading weight matrix...')
            if os.path.isfile(f + '.weights.npy'):
                weights = np.load(f + '.weights.npy')
            elif P.is_full_matrix():
                matrix = P.edge_weights[1:] if len(P.edge_weights[0]) == 1 else P.edge_weights
                weights = np.array(sum(matrix, []), dtype=int).reshape((N, N))
            else:
                print('creating weight matrix...')
                weights = np.zeros((N, N), dtype=int)
                nodes = list(P.get_nodes())
                for i in range(N):
                    for j in range(N):
                        weights[i, j] = P.get_weight(nodes[i], nodes[j])

            print(f'starting solving {f} ...')
            (paths, lengths, lb, ub), seconds = solve_mtsp(start_positions=[0], end_positions=[0], weights=weights, timeout=timeout_ms)
            if paths is None:
                print('solve_mtsp error:', lb)
                result_string = f'{base_name:<15s} N={N:>5d} A=1 mode=sum time=-------s result=-------- gap=-------\n'
            else:
                if lb > best_lb or ub < best_ub:
                    print(f'ERROR in {base_name}: bounds are [{lb}, {ub}] but best known bounds are [{best_lb}, {best_ub}]. Aborting...')
                    sys.exit(1)
                if lb >= ub and ub != best_ub:
                    print(f'ERROR in {base_name}: found solution {ub} but known solution is {best_ub}. Aborting...')
                    sys.exit(1)
                gap = ub / lb - 1 if lb > 0 else float('inf')
                result_string = f'{base_name:<15s} N={N:>5d} A=1 mode=sum time={seconds:>7.3f}s result={lengths[0]:>8d} gap={gap:>7.2%}\n'
        else:
            result_string = f'{base_name:<15s} N={N:>5d} A=1 mode=sum time=-------s result=-------- gap=-------\n'
    
        print(result_string)
        with open(bench_file, 'a') as f:
            f.write(result_string)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
