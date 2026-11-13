"""Tests for experimental structural arithmetic parsing."""

from __future__ import annotations

from sympy import Add, Mul, Pow, S, latex, sstr, symbols
from sympy.parsing.structural import Div, parse_structural_expr, touch


def test_structural_division():
    a, b, x = symbols('a b x')
    assert parse_structural_expr('2/8') == Div(2, 8)
    assert parse_structural_expr('a/b') == Div(a, b)
    assert parse_structural_expr('a*b**-1') == Mul(
        a, Pow(b, -1, evaluate=False), evaluate=False)
    assert parse_structural_expr('a/b') != parse_structural_expr('a*b**-1')
    assert touch(Div(2, 8)) == S(1)/4
    assert Div(2, 8).doit() == S(1)/4
    assert Div(x + 2, 3).args == (x + 2, S(3))
    assert latex(Div(2, 8)) == r'\frac{2}{8}'


def test_structural_multiplication_order():
    x = symbols('x')
    first = parse_structural_expr('(1/3)*(x+2)*x')
    second = parse_structural_expr('1/3*x*(x+2)')
    x2 = Add(x, 2, evaluate=False)
    assert first == Mul(Mul(Div(1, 3), x2, evaluate=False), x,
                        evaluate=False)
    assert second == Mul(Mul(Div(1, 3), x, evaluate=False), x2,
                         evaluate=False)
    assert first != second
    assert touch(first) == x*(x/3 + S(2)/3)
    assert touch(second) == x*(x + 2)/3
    assert sstr(first, order='none') == '((1/3)*(x + 2))*x'
    assert sstr(second, order='none') == '((1/3)*x)*(x + 2)'


def test_structural_parenthesized_operands():
    x, y, z = symbols('x y z')
    assert parse_structural_expr('3*(x+1)') == Mul(3, Add(x, 1, evaluate=False), evaluate=False)
    assert parse_structural_expr('x*(x+2)/3') == Div(Mul(x, Add(x, 2, evaluate=False), evaluate=False), 3)
    assert parse_structural_expr('(x*y)*z') == Mul(Mul(x, y, evaluate=False), z, evaluate=False)
    assert parse_structural_expr('x*(y*z)') == Mul(x, Mul(y, z, evaluate=False), evaluate=False)
    assert touch(parse_structural_expr('x*(x+2)/3')) == x*(x + 2)/3
    assert sstr(parse_structural_expr('(x*y)*z'), order='none') == '(x*y)*z'
    assert sstr(parse_structural_expr('x*(y*z)'), order='none') == 'x*(y*z)'


def test_structural_implicit_multiplication():
    x = symbols('x')
    assert parse_structural_expr(
        '2x', transformations='all') == Mul(2, x, evaluate=False)


def test_structural_division_parentheses():
    a, b, c = symbols('a b c')
    assert sstr(parse_structural_expr('a/(b*c)'), order='none') == 'a/(b*c)'
    assert sstr(parse_structural_expr('a*(b/c)'), order='none') == 'a*(b/c)'
    assert sstr(parse_structural_expr('(a/b)*c'), order='none') == '(a/b)*c'
    assert sstr(parse_structural_expr('(a*b)/c'), order='none') == '(a*b)/c'
    assert sstr(parse_structural_expr('a/(b/c)'), order='none') == 'a/(b/c)'
    assert latex(Pow(Div(a, b), 2, evaluate=False)) == (
        r'\left(\frac{a}{b}\right)^{2}')
    assert touch(parse_structural_expr('a/(b/c)')) == a*c/b


def test_division_reciprocal_operand_printing():
    a, b = symbols('a b')
    assert sstr(Div(a, S(1)/3), order='none') == 'a/(1/3)'
    assert sstr(Div(S(1)/3, a), order='none') == '(1/3)/a'
    assert sstr(Div(a, Pow(b, -1, evaluate=False)), order='none') == 'a/(1/b)'
    assert sstr(Div(Pow(b, -1, evaluate=False), a), order='none') == '(1/b)/a'
    assert touch(Div(a, S(1)/3)) == 3*a
    assert touch(Div(a, Pow(b, -1, evaluate=False))) == a*b
