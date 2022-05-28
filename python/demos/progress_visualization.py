import os
import tsplib95 as tsplib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtsp_vrp import solve_mtsp_vrp


def draw_fractional_solution(fractional_values, node_coords, name):
    epsilon = 1e-10
    A, N, _ = np.shape(fractional_values)
    agents, s_ids, t_ids = np.where(fractional_values > epsilon)
    values = fractional_values[agents, s_ids, t_ids]
    s_ids[s_ids == N-1] = 0 # TODO: This is a hack. Resolve this issue correctly
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


instance = 'berlin52.tsp'
base = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(base, '..', 'tsplib', 'tsp', instance)

P = tsplib.load(file)
N = P.dimension

weights = np.load(file + '.weights.npy')

fractionals = []
def callback(fractional):
    fractionals.append(fractional)

solve_mtsp_vrp(start_positions=[0], end_positions=[0], weights=weights, timeout=60000, number_of_threads=1, fractional_callback=callback)

for i, fractional in enumerate(tqdm(fractionals)):
    draw_fractional_solution(fractional, P.node_coords, f'{instance}_{i}.png')
