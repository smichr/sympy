"""Benchmarks for experimental lazy polynomial term streams."""

from sympy import Add, Mul, Poly, Pow, binomial, symbols
from sympy.polys.termstream import PolyTermStream


x = symbols('x')


def finite_difference_power(n, k):
    """Return the k-th forward difference of x**n, unevaluated."""
    return Add(*(
        (-1)**(k - j)*binomial(k, j)*Pow(x + j, n, evaluate=False)
        for j in range(k + 1)
    ), evaluate=False)


power_1000 = Pow(x + 1, 1000, evaluate=False)
finite_difference_300_10 = finite_difference_power(300, 10)
finite_difference_100000_10 = finite_difference_power(100000, 10)

factorized_100 = Add(
    Mul(
        Pow(x + 1, 100, evaluate=False),
        Pow(x - 1, 100, evaluate=False),
        evaluate=False,
    ),
    -Pow(x**2 - 1, 100, evaluate=False),
    x,
    evaluate=False,
)

finite_difference_depths = {
    k: finite_difference_power(1000, k)
    for k in (1, 2, 5, 10, 20)
}


def timeit_termstream_power_1000_degree():
    PolyTermStream(power_1000, x).degree()


def timeit_poly_power_1000_degree():
    Poly(power_1000, x).degree()


def timeit_termstream_finite_difference_300_10_degree():
    PolyTermStream(finite_difference_300_10, x).degree()


def timeit_poly_finite_difference_300_10_degree():
    Poly(finite_difference_300_10, x).degree()


def timeit_termstream_factorized_identity_100_degree():
    PolyTermStream(factorized_100, x).degree()


def timeit_poly_factorized_identity_100_degree():
    Poly(factorized_100, x).degree()


def timeit_termstream_finite_difference_100000_10_degree():
    PolyTermStream(finite_difference_100000_10, x).degree()


def timeit_termstream_finite_difference_depth_1():
    PolyTermStream(finite_difference_depths[1], x).degree()


def timeit_termstream_finite_difference_depth_2():
    PolyTermStream(finite_difference_depths[2], x).degree()


def timeit_termstream_finite_difference_depth_5():
    PolyTermStream(finite_difference_depths[5], x).degree()


def timeit_termstream_finite_difference_depth_10():
    PolyTermStream(finite_difference_depths[10], x).degree()


def timeit_termstream_finite_difference_depth_20():
    PolyTermStream(finite_difference_depths[20], x).degree()


def termstream_finite_difference_stats(k):
    """Return work counters for a cancellation-depth experiment."""
    stream = PolyTermStream(
        finite_difference_depths[k], x, collect_stats=True)
    stream.degree()
    return stream.stats
