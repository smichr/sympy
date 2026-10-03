from sympy import Add, Mul, Poly, Pow, S, symbols

from sympy.polys.termstream import (
    PolyTermStream, _polynomial_product_data,
)


x = symbols('x')
a = symbols('a')


def assert_stream_equal(got, expected):
    assert len(got) == len(expected)
    for (degree, coeff), (expected_degree, expected_coeff) in \
            zip(got, expected):
        assert degree == expected_degree
        assert coeff.equals(expected_coeff) is True


def _poly_terms(expr):
    poly = Poly(expr.doit().expand(), x)
    if poly.is_zero:
        return []
    return [(monom[0], coeff) for monom, coeff in poly.terms()]


def test_termstream_affine_product_three_factors():
    factors = [
        Pow(2*x + 3, 4, evaluate=False),
        Pow(5*x - 7, 3, evaluate=False),
        Pow(3*x + 1, 2, evaluate=False),
    ]
    expr = Mul(*factors, evaluate=False)

    expected = _poly_terms(expr)
    stream = PolyTermStream(expr, x, collect_stats=True)

    assert stream.take(len(expected) + 1) == expected
    assert stream.stats['mul_states_popped'] == 0


def test_termstream_affine_product_four_factors():
    factors = [
        Pow(x + 1, 5, evaluate=False),
        Pow(2*x - 3, 4, evaluate=False),
        Pow(3*x + 2, 3, evaluate=False),
        Pow(5*x - 1, 2, evaluate=False),
    ]
    expr = Mul(*factors, evaluate=False)

    expected = _poly_terms(expr)
    stream = PolyTermStream(expr, x, collect_stats=True)

    assert stream.take(len(expected) + 1) == expected
    assert stream.stats['mul_states_popped'] == 0


def test_termstream_affine_product_avoids_multidimensional_convolution():
    factors = [
        Pow(2*x + 3, 8, evaluate=False),
        Pow(5*x - 7, 6, evaluate=False),
        Pow(3*x + 1, 5, evaluate=False),
    ]
    expr = Mul(*factors, evaluate=False)
    stream = PolyTermStream(expr, x, collect_stats=True)

    assert stream.take(8) == _poly_terms(expr)[:8]
    assert stream.stats['mul_states_popped'] == 0


def test_termstream_symbolic_product_falls_back_to_convolution():
    expr = Mul(
        Pow(a*x + 3, 8, evaluate=False),
        Pow(5*x - 7, 6, evaluate=False),
        evaluate=False,
    )
    stream = PolyTermStream(expr, x, collect_stats=True)

    assert stream.take(8) == _poly_terms(expr)[:8]
    assert stream.stats['mul_states_popped'] > 0


def test_termstream_polynomial_power_recurrence():
    expressions = [
        Pow(x**2 + x + 1, 8, evaluate=False),
        Pow(2*x**2 + 3*x + 5, 6, evaluate=False),
        Pow(x**3 - x + 2, 5, evaluate=False),
    ]

    for expr in expressions:
        expected = _poly_terms(expr)
        stream = PolyTermStream(expr, x, collect_stats=True)
        assert stream.take(len(expected) + 1) == expected
        assert stream.stats['mul_states_popped'] == 0


def test_termstream_product_of_polynomial_powers():
    factors = [
        Pow(x**2 + x + 1, 4, evaluate=False),
        Pow(2*x + 3, 3, evaluate=False),
        Pow(x**3 - x + 2, 2, evaluate=False),
    ]
    expr = Mul(*factors, evaluate=False)

    expected = _poly_terms(expr)
    stream = PolyTermStream(expr, x, collect_stats=True)

    assert stream.take(len(expected) + 1) == expected
    assert stream.stats['mul_states_popped'] == 0


def test_termstream_symbolic_polynomial_power_falls_back():
    expr = Pow(a*x**2 + x + 1, 10, evaluate=False)
    stream = PolyTermStream(expr, x, collect_stats=True)

    assert_stream_equal(stream.take(10), _poly_terms(expr)[:10])
    assert stream.stats['mul_states_popped'] > 0


def test_termstream_product_decomposition_combines_repeated_bases():
    base = x**2 + x + 1
    expr = Mul(
        Pow(base, 4, evaluate=False),
        Pow(base, 5, evaluate=False),
        evaluate=False,
    )

    scale, degree_shift, factors = _polynomial_product_data(expr, x)
    assert scale == 1
    assert degree_shift == 0
    assert len(factors) == 1
    assert factors[0][1:] == (2, 9)

    stream = PolyTermStream(expr, x, collect_stats=True)
    assert stream.take(10) == _poly_terms(expr)[:10]
    assert stream.stats['mul_states_popped'] == 0


def test_termstream_factor_terms_exposes_common_factors():
    expr = Mul(
        Pow(Add(x, x**2, evaluate=False), 2, evaluate=False),
        Pow(1 + x, 2, evaluate=False),
        evaluate=False,
    )

    scale, degree_shift, factors = _polynomial_product_data(expr, x)
    assert scale == 1
    assert degree_shift == 2
    assert len(factors) == 1
    assert factors[0][1:] == (1, 4)

    stream = PolyTermStream(expr, x, collect_stats=True)
    assert stream.take(10) == _poly_terms(expr)
    assert stream.stats['mul_states_popped'] == 0


def test_termstream_monomial_factor_is_degree_shift():
    expr = Mul(
        Pow(x, 100000, evaluate=False),
        Pow(x + 1, 4, evaluate=False),
        evaluate=False,
    )
    stream = PolyTermStream(expr, x, collect_stats=True)

    assert stream.take(10) == [
        (100004, 1),
        (100003, 4),
        (100002, 6),
        (100001, 4),
        (100000, 1),
    ]
    assert stream.stats['mul_states_popped'] == 0
