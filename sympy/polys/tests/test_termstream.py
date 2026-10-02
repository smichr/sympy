from sympy import S, binomial, symbols
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.termstream import PolyTermStream
from sympy.testing.pytest import raises


x = symbols('x')


def test_termstream_power():
    n = 100000
    stream = PolyTermStream((x + 1)**n, x)

    assert stream.take(3) == [
        (n, 1),
        (n - 1, n),
        (n - 2, binomial(n, 2)),
    ]
    assert stream.degree() == n
    assert stream.LC() == 1


def test_termstream_add_cancellation():
    n = 100000
    stream = PolyTermStream((x + 1)**n - (x - 1)**n, x)

    assert stream.take(3) == [
        (n - 1, 2*n),
        (n - 3, 2*binomial(n, 3)),
        (n - 5, 2*binomial(n, 5)),
    ]
    assert stream.degree() == n - 1


def test_termstream_mul():
    stream = PolyTermStream((x + 1)*(x**2 + 2), x)

    assert stream.take(5) == [
        (3, 1),
        (2, 1),
        (1, 2),
        (0, 2),
    ]


def test_termstream_mul_cancellation():
    stream = PolyTermStream((x + 1)*(x - 1) - x**2, x)

    assert stream.take(2) == [(0, -1)]
    assert stream.degree() == 0
    assert stream.LC() == -1


def test_termstream_exhausted_by_cancellation():
    expr = (x + 1)**2 - (x**2 + 2*x + 1)
    stream = PolyTermStream(expr, x)

    assert stream.take(1) == []
    assert stream.degree() is S.NegativeInfinity
    assert stream.LC() == 0


def test_termstream_nested_power():
    expr = ((x + 1)**1000 + (x - 1)**1000)**100
    stream = PolyTermStream(expr, x)

    assert stream.degree() == 100000
    assert stream.LC() == 2**100
    assert stream.take(2)[1][0] == 99998


def test_termstream_rejects_nonpolynomial_nodes():
    from sympy import sin

    raises(PolynomialError, lambda: PolyTermStream(sin(x), x))
    raises(PolynomialError, lambda: PolyTermStream(1/x, x))
