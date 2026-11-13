"""Tests for experimental structural arithmetic parsing."""

from __future__ import annotations

from sympy import Add, Integral, Mul, Pow, S, UnevaluatedExpr, sin, sstr, symbols
from sympy.parsing.structural import parse_structural_expr, touch


def test_inverse_distinction():
    a, b = symbols('a b')
    div = parse_structural_expr('a/b')
    power = parse_structural_expr('a*b**-1')
    assert div == Mul(a, Pow(b, -1, evaluate=False), evaluate=False)
    assert power == Mul(
        a, Pow(b, UnevaluatedExpr(-1), evaluate=False), evaluate=False)
    assert div != power
    assert sstr(div, order='none') == 'a/b'
    assert sstr(power, order='none') == 'a*b**(-1)'
    assert touch(div) == touch(power) == a/b
    assert parse_structural_expr('b**-1') == Pow(
        b, UnevaluatedExpr(-1), evaluate=False)
    assert touch(parse_structural_expr('b**-1')) == 1/b
    assert sstr(parse_structural_expr('b**-2'), order='none') == 'b**(-2)'
    assert touch(parse_structural_expr('b**-2')) == b**-2


def test_literals():
    a = symbols('a')
    assert touch(parse_structural_expr('2/8')) == S(1)/4
    assert touch(parse_structural_expr('2**-2')) == S(1)/4
    assert sstr(parse_structural_expr('2/8'), order='none') == '2/8'
    assert sstr(parse_structural_expr('2**-2'), order='none') == '2**(-2)'
    assert touch(parse_structural_expr('a/(1/3)')) == 3*a
    assert touch(parse_structural_expr('(1/3)/a')) == 1/(3*a)


def test_multiplication_order():
    x = symbols('x')
    first = parse_structural_expr('(1/3)*(x+2)*x')
    second = parse_structural_expr('1/3*x*(x+2)')
    x2 = Add(x, 2, evaluate=False)
    inverse3 = Mul(1, Pow(3, -1, evaluate=False), evaluate=False)
    assert first == Mul(Mul(inverse3, x2, evaluate=False), x,
                        evaluate=False)
    assert second == Mul(Mul(inverse3, x, evaluate=False), x2,
                         evaluate=False)
    assert first != second
    assert sstr(first, order='none') == '((1/3)*(x + 2))*x'
    assert sstr(second, order='none') == '((1/3)*x)*(x + 2)'
    assert touch(first) == x*(x/3 + S(2)/3)
    assert touch(second) == x*(x + 2)/3


def test_nested_expressions():
    a, b, c, x, y, z = symbols('a b c x y z')
    assert parse_structural_expr('x*(y*z)') == Mul(
        x, Mul(y, z, evaluate=False), evaluate=False)
    assert parse_structural_expr('(x*y)*z') == Mul(
        Mul(x, y, evaluate=False), z, evaluate=False)
    assert sstr(parse_structural_expr('x*(y*z)'), order='none') == 'x*(y*z)'
    assert sstr(parse_structural_expr('(x*y)*z'), order='none') == '(x*y)*z'
    assert touch(parse_structural_expr('x*(x+2)/3')) == x*(x + 2)/3
    assert touch(parse_structural_expr('a/(b/c)')) == a*c/b
    assert touch(parse_structural_expr('a/(b**-1)')) == a*b
    assert sstr(parse_structural_expr('a*(b/c)'), order='none') == 'a*(b/c)'
    assert sstr(parse_structural_expr('(a/b)*c'), order='none') == '(a/b)*c'
    assert sstr(parse_structural_expr('(a*b)/c'), order='none') == '(a*b)/c'


def test_non_arithmetic():
    x = symbols('x')
    e = sin(Pow(x, UnevaluatedExpr(-1), evaluate=False))
    assert touch(e) == sin(1/x)
    e = Integral(Pow(x, UnevaluatedExpr(-1), evaluate=False), (x, 1, 2))
    assert touch(e) == Integral(1/x, (x, 1, 2))
    assert touch(e) != e.doit()


def test_roundtrip():
    for src in ['a/b', 'a*b**-1', 'a*b**(-2)', 'a/(b*c)',
                '(a/b)*c', '(a*b)/c', 'a*(b/c)',
                'x*(y*z)', '(x*y)*z', '2/8', 'x**-1', '1/3*x*(x+2)']:
        e = parse_structural_expr(src)
        p = sstr(e, order='none')
        assert parse_structural_expr(p) == e, (src, p)


def test_structural_implicit_multiplication():
    x = symbols('x')
    assert parse_structural_expr(
        '2x', transformations='all') == Mul(2, x, evaluate=False)
