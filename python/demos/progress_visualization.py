import os
import sys
import tsplib95 as tsplib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtsp_vrp import solve_mtsp_vrp


def draw_fractional_solution(fractional_values, weights, mode, start_positions, end_positions, node_coords, name):
    epsilon = 1e-10
    colors = list(mcolors.BASE_COLORS.keys())

    # delete artificial connections between end and start nodes
    for e, s in zip(end_positions, start_positions[1:] +[start_positions[0]]):
        fractional_values[:, e, s] = 0

    a_ids, s_ids, t_ids = np.where(fractional_values > epsilon)
    values = fractional_values[a_ids, s_ids, t_ids]

    names, points = zip(*node_coords.items())
    X, Y = zip(*points)
    X, Y = np.array(X, dtype='float64'), np.array(Y, dtype='float64')

    X -= np.min(X)
    Y -= np.min(Y)
    m = max(np.max(X), np.max(Y))
    X *= 100 / m
    Y *= 100 / m

    plt.switch_backend('Agg')
    plt.figure(figsize=(20, 20))

    mode_func = getattr(np, mode)

    plt.annotate(f'objective: {mode_func(np.sum(weights * fractional_values, axis=(1,2)))}', (90, 90))

    plt.plot(X, Y, '.')
    for s, e, c in zip(start_positions, end_positions, colors):
        plt.plot(X[s], Y[s], 'o', c=c)
        plt.plot(X[e], Y[e], '>', c=c)
    for x, y, n in zip(X, Y, names):
        plt.annotate(str(n), (x, y))

    for a, s, t, v in zip(a_ids, s_ids, t_ids, values):
        xx = X[[s, t]]
        yy = Y[[s, t]]
        style = '-' if v > 1 - epsilon else ':'
        plt.plot(xx, yy, style, c=colors[a])
        plt.annotate(f'{v:3.2f}', (np.mean(xx), np.mean(yy)), c=colors[a])

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.axis('square')
    plt.grid()
    plt.savefig(name)


skips = {'sum': 1, 'max': 100}

mode = sys.argv[1]
skip = skips[mode]
instance = 'berlin52.tsp'
base = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(base, '..', 'tsplib', 'tsp', instance)

P = tsplib.load(file)
N = P.dimension

weights = np.load(file + '.weights.npy')

fractionals = []
i = 0
def callback(fractional):
    global i
    fractionals.append(fractional)
    if i % skip == 0:
        fractionals.append(fractional)
    i += 1

sp = [0, 0]
ep = [0, 0]
solve_mtsp_vrp(start_positions=sp, end_positions=ep, weights=weights, optimization_mode=mode, timeout=60000, fractional_callback=callback, number_of_threads=1)

for i, fractional in enumerate(tqdm(fractionals)):
    draw_fractional_solution(fractional, weights, mode, sp, ep, P.node_coords, f'{instance}_{mode}_{i*skip}.png')
