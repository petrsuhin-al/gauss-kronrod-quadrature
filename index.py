from math import cos, sin
from nodes_count_enum import NodesCountEnum
from datetime import datetime
from scipy.integrate import quad
from gauss_kronrod import GaussKronrodQuadrature
import time

p = 100
a, b = 1, 4
nodes = NodesCountEnum.SIXTY_ONE_NODES.value


def f(x): return x * sin(p * x)
def g(x): return -x / p * cos(p * x) + 1 / p ** 2 * sin(p * x)


if __name__ == "__main__":
    start_time = datetime.now()

    expected = g(b) - g(a)

    gauss_kronrod = GaussKronrodQuadrature()
    gauss_kronrod.set_nodes(nodes)

    for result, esterror in (quad(f, a, b), gauss_kronrod.integrate(f, a, b)):
        print("{:15.13f} {:15g} {:15g}".format(result, esterror, 1 - result / expected))

    time.sleep(5)
    print('TIME:', datetime.now() - start_time)
