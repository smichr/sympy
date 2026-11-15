"""Experiments for persistent subtraction on the molar branch (gh-4596)."""
from sympy import Mul, S, UnevaluatedExpr, latex, symbols
from sympy.core.neg import Neg


def test_subtract_and_ordinary_negate_are_distinct():
    a, b, x, y = symbols('a b x y')
    assert x - x == 0
    assert x - 2*x == -x
    assert a - (x + y) == a - x - y
    assert a + -(x + y) == a + Mul(-1, x + y)
    assert a + Mul(-1, x + y) != a - (x + y)
    assert Neg(x + y) == Neg(x) + Neg(y)


def test_substitution_preserves_subtraction():
    a, b, n, x, y = symbols('a b n x y')
    assert (a - b).subs(b, x + y) == a - (x + y)
    assert (a - n*(x + y)).subs(n, S.One) == a - (x + y)
    assert (a + n*(x + y)).subs(n, -1) == a + Mul(-1, x + y)
    assert a - 1*(x + y) == a - x - y


def test_group_prevents_distributing_subtraction():
    a, x, y = symbols('a x y')
    G = UnevaluatedExpr
    e = a - G(x + y)
    assert str(e) == 'a - (x + y)'
    assert latex(e) == r'a - \left(x + y\right)'
    assert e.doit() == a - x - y
    assert G(6)*(x + 1) != 6*(x + 1)


def test_neg_sign_and_derivative():
    x = symbols('x')
    p = symbols('p', positive=True)
    assert Neg(p).is_negative is True
    assert (3 - x).diff(x) == -1
