from consts import gauss_weights, kronrod_weights, gauss_kronrod_nodes
from nodes_count_enum import NodesCountEnum
from scipy.integrate import quad
from math import cos, sin, sqrt
from bisect import insort
import numpy as np
import ctypes
import multiprocessing as mp


def get_current_kronrod_weight(nodes_count):
    return np.append(kronrod_weights[nodes_count], kronrod_weights[nodes_count][:-1][::-1])


def get_current_gauss_kronrod_nodes(nodes_count):
    return np.append(
        gauss_kronrod_nodes[nodes_count],
        np.negative(gauss_kronrod_nodes[nodes_count][:-1][::-1])
    )


def get_current_gauss_weight(nodes_count):
    changedGaussWeightArr = gauss_weights[nodes_count][:-1][::-1] \
        if divmod(nodes_count, 10)[0] % 2 \
        else gauss_weights[nodes_count][::-1]
    changedGaussWeightArr = np.append(gauss_weights[nodes_count], changedGaussWeightArr)

    for item in range(len(changedGaussWeightArr)):
        index = int(item + item + 1)

        changedGaussWeightArr = np.concatenate((changedGaussWeightArr[:index], [0], changedGaussWeightArr[index:]))

    return np.append([0], changedGaussWeightArr)


def integrate_gausskronrod(f, a, b, nodes, args=()):
    available_kronrod_weights = get_current_kronrod_weight(nodes)
    available_gauss_weights = get_current_gauss_weight(nodes)
    available_gauss_kronrod_nodes = get_current_gauss_kronrod_nodes(nodes)

    assert b > a

    mid = 0.5 * (b + a)
    dx = 0.5 * (b - a)
    zi = mid + available_gauss_kronrod_nodes * dx

    integrand = f(zi)

    integral_G = np.sum(integrand * available_gauss_weights)
    integral_K = np.sum(integrand * available_kronrod_weights)

    error = (200 * abs(integral_G - integral_K)) ** 1.5

    return integral_K * dx, dx * error, a, b


def integrate(f, a, b, nodes, args=(), min_intervals=2, limit=200, tol=1e-10):
    fv = np.vectorize(f)
    # print(f)
    intervals = []

    limits = np.linspace(a, b, min_intervals + 1)

    # for left, right in zip(limits[:-1], limits[1:]):
    #     I, err, a, b = integrate_gausskronrod(fv, left, right, nodes, args)
    #     insort(intervals, (err, a, b, I))

    data = []
    for left, right in zip(limits[:-1], limits[1:]):
        data.append((left, right))

    with mp.Pool(mp.cpu_count()) as pool:
        results = [pool.apply(
            integrate_gausskronrod,
            args=[fv, left, right, nodes, args]
        ) for left, right in zip(limits[:-1], limits[1:])]

    for I, err, left, right in results:
        insort(intervals, (err, left, right, I))

    while True:
        Itotal = sum([x[3] for x in intervals])
        err2 = sum([x[0] ** 2 for x in intervals])
        err = sqrt(err2)

        if abs(err / Itotal) < tol:
            return Itotal, err

        # нет сходимости
        if len(intervals) >= limit:
            return False

        err, left, right, I = intervals.pop()

        # разделяем интеграл
        mid = left + (right - left) / 2

        # вычисляем интегралы и ошибки, заменяем один элемент в списке и добавляем другой в конец
        I, err, a, b = integrate_gausskronrod(fv, left, mid, nodes, args)
        insort(intervals, (err, left, mid, I))
        I, err, a, b = integrate_gausskronrod(fv, mid, right, nodes, args)
        insort(intervals, (err, mid, right, I))


def f(x):
    p = 100
    return x * sin(p * x)

if __name__ == "__main__":
    manager = mp.Manager()

    p = 100
    # f = lambda x: x * sin(p * x)
    g = lambda x: -x / p * cos(p * x) + 1 / p ** 2 * sin(p * x)
    a, b = 1, 4
    nodes = NodesCountEnum.FIFTEEN_NODES.value

    expected = g(b) - g(a)

    for result, esterror in (quad(f, a, b), integrate(f, a, b, nodes)):
        print("{:15.13f} {:15g} {:15g}".format(result, esterror, 1 - result / expected))