from __future__ import annotations

from math import isqrt
from unittest.mock import patch

from sympy.core.add import Add
from sympy.core.numbers import I, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.numberfields import _algebraic_zero
from sympy.polys.numberfields._algebraic_zero import _is_algebraic_zero, _relation_zero


def _radical_residual():
    a = ((sqrt(6913) - 1)/16)**Rational(1, 3)
    b = sqrt(-6/a + 2*a + Rational(17, 4))
    c = sqrt(6/a - 2*a + Rational(17, 2) + 121/(4*b))
    t = b/2 - c/2 + Rational(9, 4)
    return t + (t**3 + 1)**Rational(1, 4) - 2


def test_algebraic_zero_nested_radicals():
    r = _radical_residual()
    assert _is_algebraic_zero(r) is True
    assert _is_algebraic_zero(r + Rational(1, 10**20)) is False
    assert _is_algebraic_zero(r - Rational(1, 10**100)) is False
    assert _is_algebraic_zero(r + Rational(1, 10**200)) is False
    assert r.is_zero is True


def test_algebraic_zero_certificates():
    a = sqrt(2)
    assert _is_algebraic_zero(1/(a - 1) - a - 1) is True
    assert _is_algebraic_zero(sqrt(1/(a + 1)) - sqrt(a - 1)) is True
    assert _is_algebraic_zero(1/(a + I) - (a - I)/3) is True
    assert _is_algebraic_zero(sqrt(2) + I*sqrt(3) - 1) is False
    # A polynomial relation for a cube root also allows other branches.
    assert _is_algebraic_zero((-S.One)**Rational(1, 3) + 1) is None
    t = 3**Rational(1, 3) + 3
    assert _is_algebraic_zero((t**3).expand()**Rational(1, 3) - t) is True
    assert _is_algebraic_zero((t**3).expand()**Rational(1, 3) + t) is False
    # Removing a radical from an unevaluated Add removes only one instance.
    r = Pow(2, Rational(1, 2), evaluate=False)
    assert _is_algebraic_zero(Add(r, r, -r, evaluate=False)) is False


def test_algebraic_zero_unknown():
    assert _is_algebraic_zero(sin(1)) is None
    assert _is_algebraic_zero(Symbol('x')) is None
    assert _is_algebraic_zero(S.Half.evalf()) is None
    assert _is_algebraic_zero(Pow(0, -1, evaluate=False)) is None
    assert _is_algebraic_zero(2**Rational(1, 17)) is None


def test_algebraic_zero_close_rationals():
    for p in (200, 500, 1000):
        q = 10**p
        r = Rational(isqrt(2*q*q), q) - sqrt(2)
        assert _is_algebraic_zero(r) is False
        assert r.is_zero is False


def test_algebraic_zero_rank_and_budget():
    # The first prime cannot invert this rational denominator.
    assert _relation_zero(sqrt(2)/65521 + I) is False
    # A singular multiplication matrix cannot identify a principal branch.
    assert _relation_zero(Pow(-8, Rational(1, 3), evaluate=False) + 2) is None
    with patch.object(_algebraic_zero, '_MAX_WORK', 0):
        assert _is_algebraic_zero(sqrt(2) + I*sqrt(3) - 1) is None
