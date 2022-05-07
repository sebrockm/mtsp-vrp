import tsplib95 as tsplib
import numpy as np
from numpy.ctypeslib import as_array
from ctypes import *
import json
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from mtsp_vrp import solve_mtsp_vrp


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()

        return ret, time2 - time1
    return wrap

def draw_fractional_solution(fractional_values, node_coords, name):
    epsilon = 1e-10
    A, N, _ = np.shape(fractional_values)
    agents, s_ids, t_ids = np.where(fractional_values > epsilon)
    values = fractional_values[agents, s_ids, t_ids]
    s_ids[s_ids == N-1] = 0
    t_ids[t_ids == N-1] = 0

    names, points = zip(*node_coords.items())
    X, Y = zip(*points)
    X, Y = np.array(X), np.array(Y)

    X -= np.min(X)
    Y -= np.min(Y)
    m = max(np.max(X), np.max(Y))
    X *= 100 / m
    Y *= 100 / m

    plt.switch_backend('Agg')
    plt.figure(figsize=(20, 20))

    plt.plot(X, Y, '.')
    for p, n in zip(points, names):
        plt.annotate(str(n), p)

    for s, t, v in zip(s_ids, t_ids, values):
        xx = X[[s, t]]
        yy = Y[[s, t]]
        if v > 1 - epsilon:
            plt.plot(xx, yy, 'g-')
        else:
            plt.plot(xx, yy, 'g:')
        plt.annotate(f'{v:3.2f}', (np.mean(xx), np.mean(yy)))

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.axis('square')
    plt.grid()
    plt.savefig(name)

@timing
def solve_mtsp(start_positions, end_positions, weights, timeout):
    A = len(start_positions)
    N = len(weights)
    start_positions = np.array(start_positions, dtype=np.uint64)
    end_positions = np.array(end_positions, dtype=np.uint64)
    weights = np.array(weights, dtype=np.int32)
    number_of_threads = 1
    lb = c_double(0)
    ub = c_double(0)
    pathsBuffer = np.zeros(shape=(N,), dtype=np.uint64)
    offsets = np.zeros(shape=(A,), dtype=np.uint64)

    fractionals = []
    @CFUNCTYPE(c_int, POINTER(c_double), c_size_t, c_size_t)
    def store_fractional_solution(fractional_values, A, N):
        fractionals.append(np.copy(as_array(fractional_values, shape=(A, N, N))))
        return 0

    result = solve_mtsp_vrp(A, N, start_positions, end_positions, weights, timeout,
                            number_of_threads, byref(lb), byref(ub), pathsBuffer, offsets, store_fractional_solution)
    if result < 0:
        print(f'error: {result}')
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

    return paths, lengths, lb.value, ub.value, fractionals

def main(timeout_ms):
    base = os.path.dirname(os.path.abspath(__file__))
    files = [os.path.join(base, kind, f) for kind in ['sop', 'atsp', 'tsp'] for f in os.listdir(os.path.join(base, kind))]

    with open(os.path.join(base, 'best-known-solutions.json')) as f:
        solutions = json.load(f)

    bench_file = os.path.join(base, 'bench.txt')
    try:
        os.remove(bench_file)
    except OSError:
        pass

    missing_best_known_solutions = ['ESC11.sop']

    progress_bar = tqdm(files)
    for f in progress_bar:
        base_name = os.path.basename(f)
        progress_bar.set_description(base_name)
        if base_name != 'berlin52.tsp':
            continue
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
            (paths, lengths, lb, ub, fractionals), seconds = solve_mtsp(start_positions=[0], end_positions=[0], weights=weights, timeout=timeout_ms)

            if hasattr(P, 'node_coords') and P.node_coords:
                for i, fractional in tqdm(list(enumerate(fractionals))):
                    draw_fractional_solution(fractional, P.node_coords, f'{base_name}_{i}.png')

            if paths is None:
                print('solve_mtsp error:', lb)
                result_string = f'{base_name:<15s} N={N:>5d} A=1 mode=sum time=-------s result=-------- gap=-------\n'
            else:
                if base_name not in missing_best_known_solutions:
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
    main(int(sys.argv[1]))
