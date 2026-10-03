from sympy import Add, Mul, Poly, Pow, S, symbols

from sympy.polys.termstream import PolyTermStream


x = symbols('x')


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
    direct = Mul(*factors, evaluate=False)
    generic = Mul(
        Add(factors[0], S.Zero, evaluate=False),
        *factors[1:],
        evaluate=False,
    )

    direct_stream = PolyTermStream(direct, x, collect_stats=True)
    generic_stream = PolyTermStream(generic, x, collect_stats=True)

    assert direct_stream.take(8) == generic_stream.take(8)
    assert direct_stream.stats['mul_states_popped'] == 0
    assert generic_stream.stats['mul_states_popped'] > 0
    assert direct_stream.stats['term_requests'] < generic_stream.stats['term_requests']
