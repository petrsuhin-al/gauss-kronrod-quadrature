from consts import gauss_weights, kronrod_weights, gauss_kronrod_nodes
from nodes_count_enum import NodesCountEnum
from scipy.integrate import quad
from math import cos, sin, sqrt
from bisect import insort
import numpy as np


def get_current_kronrod_weight(nodes_count):
    global available_kronrod_weights

    if len(available_kronrod_weights) == 0:
        available_kronrod_weights = np.append(kronrod_weights[nodes_count], kronrod_weights[nodes_count][:-1][::-1])

    return available_kronrod_weights


def get_current_gauss_kronrod_nodes(nodes_count):
    global available_gauss_kronrod_nodes

    if len(available_gauss_kronrod_nodes) == 0:
        available_gauss_kronrod_nodes = np.append(
            gauss_kronrod_nodes[nodes_count],
            np.negative(gauss_kronrod_nodes[nodes_count][:-1][::-1])
        )

    return available_gauss_kronrod_nodes


def get_current_gauss_weight(nodes_count):
    global available_gauss_weights

    if len(available_gauss_weights) == 0:
        changedGaussWeightArr = gauss_weights[nodes_count][:-1][::-1] if divmod(nodes_count, 10)[0] % 2 else gauss_weights[nodes_count][::-1]
        changedGaussWeightArr = np.append(gauss_weights[nodes_count], changedGaussWeightArr)

        for item in range(len(changedGaussWeightArr)):
            index = int(item + item + 1)

            changedGaussWeightArr = np.concatenate((changedGaussWeightArr[:index], [0], changedGaussWeightArr[index:]))

        available_gauss_weights = np.append([0], changedGaussWeightArr)

    return available_gauss_weights


def integrate_gausskronrod(f, a, b, nodes, args=()):
    assert b > a

    mid = 0.5 * (b + a)
    dx = 0.5 * (b - a)
    zi = mid + get_current_gauss_kronrod_nodes(nodes) * dx

    integrand = f(zi)

    integral_G = np.sum(integrand * get_current_gauss_weight(nodes))
    integral_K = np.sum(integrand * get_current_kronrod_weight(nodes))

    error = (200 * abs(integral_G - integral_K)) ** 1.5

    return integral_K * dx, dx * error


def integrate(f, a, b, nodes, args=(), min_intervals=1, limit=200, tol=1e-10):
    fv = np.vectorize(f)

    intervals = []

    limits = np.linspace(a, b, min_intervals + 1)

    for left, right in zip(limits[:-1], limits[1:]):
        I, err = integrate_gausskronrod(fv, left, right, nodes, args)
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
        I, err = integrate_gausskronrod(fv, left, mid, nodes, args)
        insort(intervals, (err, left, mid, I))
        I, err = integrate_gausskronrod(fv, mid, right, nodes, args)
        insort(intervals, (err, mid, right, I))


if __name__ == "__main__":
    available_kronrod_weights, available_gauss_weights, available_gauss_kronrod_nodes = [], [], []
    p = 100
    f = lambda x: x * sin(p * x)
    g = lambda x: -x / p * cos(p * x) + 1 / p ** 2 * sin(p * x)
    a, b = 1, 4
    nodes = NodesCountEnum.FIFTEEN_NODES.value

    expected = g(b) - g(a)

    for result, esterror in (quad(f, a, b), integrate(f, a, b, nodes)):
        print("{:15.13f} {:15g} {:15g}".format(result, esterror, 1 - result / expected))