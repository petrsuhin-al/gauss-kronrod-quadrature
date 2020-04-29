from consts import gauss_weights, kronrod_weights, gauss_kronrod_nodes
from math import sqrt
from bisect import insort
import numpy as np
import multiprocessing as mp


class GaussKronrodQuadrature:
    def __init__(self):
        self.KRONROD_WEIGHT = []
        self.GAUSS_WEIGHT = []
        self.GAUSS_KRONROD_NODES = []

    def set_nodes(self, nodes):
        self.KRONROD_WEIGHT = self.get_current_kronrod_weight(nodes)
        self.GAUSS_WEIGHT = self.get_current_gauss_weight(nodes)
        self.GAUSS_KRONROD_NODES = self.get_current_gauss_kronrod_nodes(nodes)

    def integrate(self, f, a, b, args=(), min_intervals=2, limit=200, tol=1e-10):
        fv = np.vectorize(f)
        intervals = []

        limits = np.linspace(a, b, min_intervals + 1)

        with mp.Pool(mp.cpu_count()) as first_pool:
            results = [
                first_pool.apply(
                    self.integrate_gausskronrod,
                    args=[fv, left, right, args]
                ) for left, right in zip(limits[:-1], limits[1:])
            ]

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
            I, err, a, b = self.integrate_gausskronrod(fv, left, mid, args)
            insort(intervals, (err, left, mid, I))
            I, err, a, b = self.integrate_gausskronrod(fv, mid, right, args)
            insort(intervals, (err, mid, right, I))

    def integrate_gausskronrod(self, f, a, b, args=()):
        assert b > a

        mid = 0.5 * (b + a)
        dx = 0.5 * (b - a)
        zi = mid + self.GAUSS_KRONROD_NODES * dx

        integrand = f(zi)

        integral_G = np.sum(integrand * self.GAUSS_WEIGHT)
        integral_K = np.sum(integrand * self.KRONROD_WEIGHT)

        error = (200 * abs(integral_G - integral_K)) ** 1.5

        return integral_K * dx, dx * error, a, b

    @classmethod
    def get_current_kronrod_weight(cls, nodes_count):
        return np.append(kronrod_weights[nodes_count], kronrod_weights[nodes_count][:-1][::-1])

    @classmethod
    def get_current_gauss_kronrod_nodes(cls, nodes_count):
        return np.append(
            gauss_kronrod_nodes[nodes_count],
            np.negative(gauss_kronrod_nodes[nodes_count][:-1][::-1])
        )

    @classmethod
    def get_current_gauss_weight(cls, nodes_count):
        changedGaussWeightArr = gauss_weights[nodes_count][:-1][::-1] \
            if divmod(nodes_count, 10)[0] % 2 else gauss_weights[nodes_count][::-1]

        changedGaussWeightArr = np.append(gauss_weights[nodes_count], changedGaussWeightArr)

        for item in range(len(changedGaussWeightArr)):
            index = int(item + item + 1)

            changedGaussWeightArr = np.concatenate((changedGaussWeightArr[:index], [0], changedGaussWeightArr[index:]))

        return np.append([0], changedGaussWeightArr)
