from itertools import combinations

from sympy import Add, sin, symbols
from sympy.polys.exponentsets import (
    ExponentRuns,
    _run_difference,
    _run_intersection,
    _run_sum,
    _zip_runs,
    exponent_runs,
)
from sympy.polys.polyerrors import PolynomialError
from sympy.testing.pytest import raises


x, y = symbols('x y')


def _run_values(runs):
    values = set()
    for start, stop, step in runs:
        values.update(range(start, stop + 1, step))
    return values


def _runs():
    for start in range(5):
        for step in range(1, 6):
            for count in range(1, 7):
                stop = start + step*(count - 1)
                yield (start, stop, step)


def _small_sets(n):
    universe = range(n)
    for size in range(1, n + 1):
        for values in combinations(universe, size):
            yield frozenset(values)


def _minkowski_values(A, B):
    return {a + b for a in A for b in B}


def _power_values(values, n):
    result = {0}
    base = set(values)

    while n:
        if n & 1:
            result = _minkowski_values(result, base)
        n >>= 1
        if n:
            base = _minkowski_values(base, base)

    return result


def _assert_disjoint(R):
    runs = R.runs
    for i, A in enumerate(runs):
        for B in runs[i + 1:]:
            assert _run_intersection(A, B) is None


def _explicit_support(expr, gen):
    if expr.is_zero is True:
        return set()
    if not expr.has(gen):
        return {0}
    if expr == gen:
        return {1}
    if expr.is_Add:
        result = set()
        for arg in expr.args:
            result |= _explicit_support(arg, gen)
        return result
    if expr.is_Mul:
        result = {0}
        for arg in expr.args:
            result = _minkowski_values(result, _explicit_support(arg, gen))
        return result
    if expr.is_Pow and expr.exp.is_Integer and expr.exp >= 0:
        return _power_values(_explicit_support(expr.base, gen), int(expr.exp))
    raise PolynomialError


def test_exponent_runs_basics():
    R = ExponentRuns([(0, 10, 1), (0, 10, 2)])
    assert R.runs == ((0, 10, 1),)
    assert R.lo == 0
    assert R.hi == 10
    assert R.size == len(R) == 11
    assert 7 in R
    assert 11 not in R
    assert ExponentRuns(R).runs == R.runs

    assert ExponentRuns([(5, 5, 7)]).runs == ((5, 5, 1),)
    assert ExponentRuns().lo is None
    assert ExponentRuns().hi is None
    assert not ExponentRuns()

    raises(ValueError, lambda: ExponentRuns([(0, 10, 3)]))
    raises(ValueError, lambda: ExponentRuns([(0, 10, 0)]))
    raises(ValueError, lambda: R**-1)


def test_zip_runs():
    cases = [
        (
            [(0, 2_000_000, 2), (1, 1_999_999, 2)],
            [(0, 2_000_000, 1)],
        ),
        (
            [(0, 3_000_000, 3),
             (1, 2_999_998, 3),
             (2, 2_999_996, 3)],
            [(0, 2_999_998, 1), (3_000_000, 3_000_000, 1)],
        ),
        (
            [(0, 4_000_000, 4), (2, 3_999_998, 4)],
            [(0, 4_000_000, 2)],
        ),
    ]

    for raw, expected in cases:
        zipped = _zip_runs(raw)
        assert zipped == expected
        assert _zip_runs(zipped) == zipped
        assert _run_values(zipped) == _run_values(raw)


def test_run_difference_exhaustive():
    runs = list(_runs())

    for A in runs:
        av = _run_values([A])
        for B in runs:
            bv = _run_values([B])
            assert _run_values(_run_difference(A, B)) == av - bv


def test_run_sum_exhaustive():
    runs = list(_runs())

    for A in runs:
        av = _run_values([A])
        for B in runs:
            bv = _run_values([B])
            AB = _run_sum(A, B)
            BA = _run_sum(B, A)
            expected = _minkowski_values(av, bv)

            assert set(AB) == expected
            assert AB.runs == BA.runs
            _assert_disjoint(AB)


def test_union_exhaustive():
    sets = list(_small_sets(6))

    for av in sets:
        A = ExponentRuns.from_values(av)
        for bv in sets:
            B = ExponentRuns.from_values(bv)
            U = A | B
            assert set(U) == set(av | bv)
            _assert_disjoint(U)


def test_minkowski_algebra():
    sets = list(_small_sets(6))

    for av in sets:
        A = ExponentRuns.from_values(av)
        for bv in sets:
            B = ExponentRuns.from_values(bv)
            AB = A + B
            assert set(AB) == _minkowski_values(av, bv)
            assert set(AB) == set(B + A)
            _assert_disjoint(AB)

    # A smaller universe is enough for the cubic associativity check.
    sets = list(_small_sets(4))
    for av in sets:
        A = ExponentRuns.from_values(av)
        for bv in sets:
            B = ExponentRuns.from_values(bv)
            for cv in sets:
                C = ExponentRuns.from_values(cv)
                expected = {
                    a + b + c
                    for a in av
                    for b in bv
                    for c in cv
                }
                assert set((A + B) + C) == expected
                assert set(A + (B + C)) == expected


def test_power_exhaustive():
    # Every nonempty subset of {0, ..., 8}, for powers 1 through 7.
    for values in _small_sets(9):
        R = ExponentRuns.from_values(values)
        for n in range(1, 8):
            powered = R**n
            assert set(powered) == _power_values(values, n)
            _assert_disjoint(powered)


def test_large_sparse_power():
    R = ExponentRuns.from_values({0, 3, 7})**1_000_000

    assert R.lo == 0
    assert R.hi == 7_000_000
    assert R.size == 6_999_986
    assert len(R.runs) == 10

    for gap in (1, 2, 4, 5, 8, 11):
        assert gap not in R
    for deficit in (1, 2, 3, 5, 6, 9, 10, 13, 17):
        assert R.hi - deficit not in R

    assert 12 in R
    assert R.hi - 18 in R
    _assert_disjoint(R)


def test_exponent_runs_expression_walker():
    expressions = [
        (1 + x**3 + x**7)**20,
        (x**4 + (x**5 + x**2 + 1)**4 + 1)**10,
        (1 + x + (1 + x**2 + x**5)**4)**10,
        (x**2 + x**5 + x**9)**7,
        y*(1 + x**2)**5,
    ]

    for expr in expressions:
        assert set(exponent_runs(expr, x)) == _explicit_support(expr, x)

    assert not exponent_runs(0, x)
    assert set(exponent_runs(y, x)) == {0}

    # Add support is structural: an unevaluated cancellation remains a
    # possible exponent until a coefficient consumer proves it vanishes.
    cancelled = Add(x, -x, evaluate=False)
    assert set(exponent_runs(cancelled, x)) == {1}

    raises(PolynomialError, lambda: exponent_runs(sin(x), x))
    raises(PolynomialError, lambda: exponent_runs(x**-1, x))


def test_summary_and_repr():
    E = ExponentRuns([
        (0, 0, 1),
        (3, 3, 1),
        (6, 10, 2),
        (12, 20, 1),
    ])

    assert E.gap_count == 7

    assert repr(E) == (
        "ExponentRuns(((0, 0, 1), (3, 3, 1), "
        "(6, 10, 2), (12, 20, 1)))"
    )

    summary = E.summary()

    assert "lo=0, hi=20" in summary
    assert "count=14, span=21, gaps=7" in summary
    assert "runs=4" in summary
    assert "support=0, 3, [6, 10; 2], [12, 20]" in summary
