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
