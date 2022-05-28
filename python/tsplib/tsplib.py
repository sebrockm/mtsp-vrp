import tsplib95 as tsplib
import numpy as np
from ctypes import *
import json
import os
import sys
import time
from tqdm import tqdm

if __name__ == '__main__':
    os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mtsp_vrp import solve_mtsp_vrp

    
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
            star_time = time.time()
            paths, lengths, lb, ub = solve_mtsp_vrp(start_positions=[0], end_positions=[0], weights=weights, timeout=timeout_ms)
            seconds = time.time() - star_time
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
