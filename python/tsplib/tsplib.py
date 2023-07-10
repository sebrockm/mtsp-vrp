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


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    bench_instances = ['ESC25.sop', 'ftv170.atsp', 'dantzig42.tsp', 'dsj1000.tsp', 'u574.tsp']
    bench_instances = [os.path.join(base, os.path.splitext(i)[1][1:], i) for i in bench_instances]
    #files = [os.path.join(base, kind, f) for kind in ['sop', 'atsp', 'tsp'] for f in sorted(os.listdir(os.path.join(base, kind)), key=str.casefold)]

    with open(os.path.join(base, 'best-known-solutions.json')) as f:
        solutions = json.load(f)

    bench_file = os.path.join(base, 'bench.json')

    missing_best_known_solutions = ['ESC11.sop']

    agent_counts = [1, 2, 4, 8]

    results = []
    with tqdm(total=len(bench_instances)*len(agent_counts)) as progress_bar:
        for instance in bench_instances:
            base_name = os.path.basename(instance)
            problem_name, ext = os.path.splitext(base_name)
            kind = ext[1:]
            if kind not in ['tsp', 'atsp', 'sop']:
                continue

            P = tsplib.load(instance)
            N = P.dimension

            if os.path.isfile(instance + '.weights.npy'):
                weights = np.load(instance + '.weights.npy')
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

            for A in agent_counts:
                progress_bar.set_description(f'Solving {base_name} with {A=}')
                if A > 1 or base_name in missing_best_known_solutions:
                    best_lb = None
                    best_ub = None
                else:
                    solution = solutions[kind][problem_name]
                    if isinstance(solution, int):
                        best_lb = solution
                        best_ub = solution
                    else:
                        best_lb, best_ub = solution

                if kind == 'sop':
                    sp = [0] * A
                    ep = [N - 1] * A
                else:
                    sp = list(range(1, A+1))
                    ep = list(range(1, A+1))

                star_time = time.time()
                try:
                    paths, lengths, lb, ub = solve_mtsp_vrp(start_positions=sp, end_positions=ep, weights=weights, timeout=5*60*1000)
                    error = None
                except Exception as e:
                    error = e.args
                seconds = time.time() - star_time
                if error is not None:
                    results.append({'name': base_name, 'N': N, 'A': A, 'mode': 'sum', 'error': error})
                else:
                    if best_lb is not None and best_ub is not None:
                        if lb > best_lb or ub < best_ub:
                            input(f'ERROR in {base_name}: bounds are [{lb}, {ub}] but best known bounds are [{best_lb}, {best_ub}]. Aborting...')
                            sys.exit(1)
                        if lb >= ub and ub != best_ub:
                            input(f'ERROR in {base_name}: found solution {ub} but known solution is {best_ub}. Aborting...')
                            sys.exit(1)
                    gap = ub / lb - 1 if lb > 0 else float('inf')
                    results.append({'name': base_name, 'N': N, 'A': A, 'mode': 'sum', 'seconds': seconds, 'lower_bound': lb, 'upper_bound': ub, 'gap': gap, 'path_lengths': lengths})

                with open(bench_file, 'w') as f:
                    json.dump(results, f, indent=4)

                progress_bar.update()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
