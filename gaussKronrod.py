from consts import gausskronrod_nodes, gauss_weights, kronrod_weights
from scipy.integrate import quad
from math import cos, sin, sqrt
import numpy as np
import bisect

def integrate_gausskronrod(f, a, b, nodes, args=()):
    assert b > a

    mid = 0.5 * (b + a)
    dx = 0.5 * (b - a)
    zi = mid + gausskronrod_nodes[nodes] * dx

    integrand = f(zi)
    integral_G7 = np.sum(integrand[:7] * gauss_weights[nodes])
    integral_K15 = np.sum(integrand * kronrod_weights[nodes])

    error = (200 * abs(integral_G7 - integral_K15)) ** 1.5

    return integral_K15 * dx, dx * error


def integrate(f, a, b, nodes, args=(), minintervals=1, limit=200, tol=1e-10):
    fv = np.vectorize(f)

    intervals = []

    limits = np.linspace(a, b, minintervals + 1)

    for left, right in zip(limits[:-1], limits[1:]):
        I, err = integrate_gausskronrod(fv, left, right, nodes, args)
        bisect.insort(intervals, (err, left, right, I))

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
        bisect.insort(intervals, (err, left, mid, I))
        I, err = integrate_gausskronrod(fv, mid, right, nodes, args)
        bisect.insort(intervals, (err, mid, right, I))


if __name__ == "__main__":
    p = 100
    f = lambda x: x * sin(p * x)
    g = lambda x: -x / p * cos(p * x) + 1 / p ** 2 * sin(p * x)
    a, b = 1, 4
    nodes = 15

    expected = g(b) - g(a)

    for result, esterror in (quad(f, a, b), integrate(f, a, b, nodes)):
        print("{:15.13f} {:15g} {:15g}".format(result, esterror, 1 - result / expected))
