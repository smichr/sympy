from math import prod
from random import Random

from sympy import (
    Add, Integer, Mul, Poly, Pow, Rational, S, binomial, expand, sin,
    cos, symbols,
)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.termstream import (
    PolyTermStream, _MulStream, _StreamStats, _stream_from_expr,
    polynomial_zero,
)
from sympy.testing.pytest import raises


x, y = symbols('x y')


def assert_stream_equal(got, expected):
    assert len(got) == len(expected)
    for (degree, coeff), (expected_degree, expected_coeff) in \
            zip(got, expected):
        assert degree == expected_degree
        assert coeff.equals(expected_coeff) is True


def _poly_terms(expr):
    """Return the canonical nonzero terms of ``Poly(expr, x)``."""
    poly = Poly(expr.doit().expand(), x)
    if poly.is_zero:
        return []
    return [(monom[0], coeff) for monom, coeff in poly.terms()]


def _assert_matches_poly(expr):
    """Compare the complete stream and its leading queries with Poly."""
    expected = _poly_terms(expr)
    stream = PolyTermStream(expr, x)

    assert_stream_equal(stream.take(len(expected) + 1), expected)

    poly = Poly(expr.doit().expand(), x)
    assert stream.degree() == poly.degree()
    assert stream.LC().equals(poly.LC())

    assert all(
        left[0] > right[0]
        for left, right in zip(expected, expected[1:])
    )
    assert all(coeff != 0 for _, coeff in expected)


def _finite_difference_power(n, k):
    """Return the k-th forward difference of x**n, kept unevaluated."""
    return Add(*(
        (-1)**(k - j)*binomial(k, j)*Pow(x + j, n, evaluate=False)
        for j in range(k + 1)
    ), evaluate=False)


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


def test_termstream_cancellation_depth():
    n = 100000
    for cancelled in (1, 2, 5, 10, 20, 50):
        prefix = Add(*(
            binomial(n, k)*x**(n - k)
            for k in range(cancelled)
        ))
        stream = PolyTermStream((x + 1)**n - prefix, x)
        assert stream.term(0) == (
            n - cancelled, binomial(n, cancelled))


def test_termstream_structural_cancellation_depth():
    n = 1000
    for k in (1, 2, 5, 10):
        stream = PolyTermStream(_finite_difference_power(n, k), x)
        expected_lc = prod(range(n - k + 1, n + 1))
        assert stream.term(0) == (n - k, expected_lc)


def test_termstream_instrumentation():
    expr = _finite_difference_power(1000, 5)
    stream = PolyTermStream(expr, x, collect_stats=True)

    assert stream.stats['root_terms_generated'] == 0
    assert stream.degree() == 995

    stats = stream.stats
    assert stats['root_terms_generated'] == 1
    assert stats['cancelled_layers'] >= 5
    assert stats['generated_terms'] > 1
    assert stats['child_requests'] > 0
    assert stats['mul_states_popped'] == 0
    assert stats['streams_created'] > 1

    generated = stats['generated_terms']
    hits = stats['cache_hits']
    assert stream.LC() == prod(range(996, 1001))
    assert stream.stats['generated_terms'] == generated
    assert stream.stats['cache_hits'] > hits

    assert PolyTermStream(expr, x).stats is None


def test_termstream_instrumentation_tracks_cancellation_depth():
    previous_cancelled = -1
    previous_generated = -1
    for k in (1, 2, 5, 10):
        stream = PolyTermStream(
            _finite_difference_power(1000, k), x, collect_stats=True)
        assert stream.degree() == 1000 - k
        stats = stream.stats
        assert stats['root_terms_generated'] == 1
        assert stats['cancelled_layers'] >= k
        assert stats['cancelled_layers'] > previous_cancelled
        assert stats['generated_terms'] > previous_generated
        assert stats['mul_states_popped'] == 0
        previous_cancelled = stats['cancelled_layers']
        previous_generated = stats['generated_terms']


def test_termstream_affine_product_recurrence():
    expressions = [
        (2*x + 3)**4*(5*x - 7)**3,
        (x + 1)**8*(x - 1)**5,
        (3*x - 2)**6*(2*x + 5)**4,
    ]
    for expr in expressions:
        _assert_matches_poly(expr)


def test_termstream_affine_product_avoids_convolution():
    # Keep this control case intentionally small; the larger comparison lives
    # in bench_termstream.py because forcing the generic convolution path is
    # deliberately expensive.
    left = Pow(2*x + 3, 8, evaluate=False)
    right = Pow(5*x - 7, 6, evaluate=False)
    direct = Mul(left, right, evaluate=False)

    direct_stream = PolyTermStream(direct, x, collect_stats=True)
    stats = _StreamStats()
    generic_stream = _MulStream(
        _stream_from_expr(left, x, stats),
        _stream_from_expr(right, x, stats),
        stats,
    )

    assert_stream_equal(direct_stream.take(5), generic_stream.take(5))
    assert direct_stream.stats['mul_states_popped'] == 0
    assert stats.mul_states_popped > 0
    assert direct_stream.stats['term_requests'] < stats.term_requests


def test_termstream_factorized_identity_cancellation():
    for n in (5, 10, 30):
        factored = Mul(
            Pow(x + 1, n, evaluate=False),
            Pow(x - 1, n, evaluate=False),
            evaluate=False,
        )
        power = Pow(x**2 - 1, n, evaluate=False)
        expr = Add(factored, -power, x, evaluate=False)
        stream = PolyTermStream(expr, x)
        assert stream.term(0) == (1, 1)
        assert stream.take(2) == [(1, 1)]


def test_termstream_mul():
    stream = PolyTermStream((x + 1)*(x**2 + 2), x)

    assert stream.take(5) == [
        (3, 1),
        (2, 1),
        (1, 2),
        (0, 2),
    ]


def test_termstream_sparse_mul():
    p = x**17 + 3*x**9 - 2*x**2 + 5
    q = x**23 - x**11 + 7*x**3 - 4
    _assert_matches_poly(p*q)


def test_termstream_mul_degree_collisions():
    p = x**4 + x**2 + 1
    q = x**4 - x**2 + 1
    _assert_matches_poly(p*q)

    p = Add(*(x**i for i in range(20)))
    q = Add(*((-1)**i*x**i for i in range(20)))
    _assert_matches_poly(p*q)


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


def test_termstream_exhausted_after_many_terms():
    p = (x + 1)**12*(x**3 - 2*x + 7)
    expr = p - expand(p)
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


def test_termstream_symbolic_coefficients():
    expr = (x + 1)**10*(1 - x + y)**2 - 3
    expected = _poly_terms(expr)
    stream = PolyTermStream(expr, x)

    assert_stream_equal(stream.take(5), expected[:5])
    _assert_matches_poly(expr)

    coeff = sin(y)**2 + (1 - sin(y)**2) - 1
    _assert_matches_poly(Mul(coeff, x**10, evaluate=False) + x)


def test_termstream_cache_is_stable():
    stream = PolyTermStream((x + 1)**30*(x**5 - x + 2), x)
    first = stream.take(3)
    more = stream.take(10)

    assert more[:3] == first
    assert stream.take(3) == first
    assert stream.take(10) == more
    assert stream.degree() == more[0][0]
    assert stream.LC() == more[0][1]


def _random_poly(rng, depth):
    """Build a deliberately unevaluated small polynomial expression tree."""
    if depth == 0:
        if rng.randrange(3) == 0:
            return x
        return Integer(rng.randint(-3, 3))

    op = rng.randrange(3)
    if op == 0:
        args = [_random_poly(rng, depth - 1)
                for _ in range(rng.randint(2, 3))]
        return Add(*args, evaluate=False)
    if op == 1:
        args = [_random_poly(rng, depth - 1)
                for _ in range(rng.randint(2, 3))]
        return Mul(*args, evaluate=False)

    base = _random_poly(rng, depth - 1)
    exponent = rng.randint(1, 5)
    return Pow(base, exponent, evaluate=False)


def test_termstream_random_against_poly():
    rng = Random(8675309)
    for _ in range(500):
        expr = _random_poly(rng, 3)
        _assert_matches_poly(expr)


def test_termstream_handpicked_against_poly():
    expressions = [
        (x + 1)**8 - (x - 1)**8,
        (2*x + 1)**9 - (2*x - 1)**9,
        (x**5 + x**2 + 1)**4,
        ((x + 1)**4 - (x - 1)**4)*(x**7 - 3*x + 1),
        (x**4 + x**2 + 1)*(x**4 - x**2 + 1),
        (x + 1)**9 - expand((x + 1)**9),
    ]
    for expr in expressions:
        _assert_matches_poly(expr)


def test_termstream_rejects_nonpolynomial_nodes():
    z = symbols('z')

    for expr in (
        sin(x),
        1/x,
        x**Rational(1, 2),
        x**z,
        (x + 1)**-2,
    ):
        raises(PolynomialError, lambda expr=expr: PolyTermStream(expr, x))

    raises(TypeError, lambda: PolyTermStream(x, x + 1))


def test_termstream_nested_unevaluated_power_exponents():
    base = Pow(x + 1, 2, evaluate=False)
    expr = Pow(base, 3, evaluate=False)

    _assert_matches_poly(expr)
    assert PolyTermStream(expr, x).degree() == 6



def test_polynomial_zero_nonzero_witness():
    expr = (x + y)**20 - (x - y)**20
    assert polynomial_zero(expr) == (
        False, ({x: 19, y: 1}, 40))


def test_polynomial_zero_by_stream_exhaustion():
    a = symbols('a')
    expr = Add(
        -a**2, -2*a, Pow(a + 1, 2, evaluate=False), -1,
        evaluate=False,
    )
    assert polynomial_zero(expr) == (True, None)


def test_polynomial_zero_unresolved_residual():
    a = symbols('a')
    coeff = sin(a)**2 + cos(a)**2 - 1
    expr = Mul(x, coeff, evaluate=False)
    assert polynomial_zero(expr) == (
        None, ({x: 1}, coeff))


def test_polynomial_zero_leading_cancellation_chain():
    n = 200
    k = 6
    cancel = Add(*(
        (-1)**(k - j)*binomial(k, j)*Pow(x + j*y, n, evaluate=False)
        for j in range(k + 1)
    ), evaluate=False)
    expr = Pow(
        x**80 - Pow(x - y, 30, evaluate=False)*(y**3 + 1),
        100,
        evaluate=False,
    )*cancel

    assert polynomial_zero(expr) == (
        False,
        ({x: 8194, y: 6}, prod(range(195, 201))),
    )


def test_possible_support():
    x = symbols('x')

    stream = PolyTermStream((1 + x**3 + x**7)**1000000, x)
    support = stream.possible_support

    assert support.lo == 0
    assert support.hi == 7000000
    assert len(support) == 6999986

    assert 7000000 in support
    assert 6999999 not in support
    assert 6999996 in support


def test_possible_support_cancellation():
    x = symbols('x')
    expr = Add(x, -x, evaluate=False)

    stream = PolyTermStream(expr, x)

    assert 1 in stream.possible_support
    assert stream.degree() is S.NegativeInfinity


def test_add_stream_uses_exponent_support():
    x = symbols('x')

    expr = (
        (x + 1)**1000 +
        (x + 1)**100 +
        (x + 1)**10 +
        1
    )

    stream = PolyTermStream(expr, x, collect_stats=True)

    assert stream.take(1) == [(1000, 1)]

    # Only the degree-1000 child needs to be inspected.  Its first
    # term is requested to activate it and its second while closing
    # the degree-1000 layer.
    assert stream.stats['child_requests'] == 2


def test_add_stream_support_bound_with_cancellation():
    x = symbols('x')

    expr = (
        (x + 1)**100 -
        (x - 1)**100 +
        (x + 1)**10
    )

    stream = PolyTermStream(expr, x)

    assert stream.take(1) == [(99, 200)]
