"""Tests for experimental structural arithmetic parsing."""

from __future__ import annotations

from sympy import Pow, S, latex, symbols
from sympy.parsing.structural import Div, Mul2, parse_structural_expr, touch


def test_structural_division():
    a, b, x = symbols('a b x')
    assert parse_structural_expr('2/8') == Div(2, 8)
    assert parse_structural_expr('a/b') == Div(a, b)
    assert parse_structural_expr('a*b**-1') == Mul2(
        a, Pow(b, -1, evaluate=False))
    assert parse_structural_expr('a/b') != parse_structural_expr('a*b**-1')
    assert touch(Div(2, 8)) == S(1)/4
    assert Div(2, 8).doit() == S(1)/4
    assert Div(x + 2, 3).args == (x + 2, S(3))
    assert latex(Div(2, 8)) == r'\frac{2}{8}'


def test_structural_multiplication_order():
    x = symbols('x')
    first = parse_structural_expr('(1/3)*(x+2)*x')
    second = parse_structural_expr('1/3*x*(x+2)')
    assert first == Mul2(Mul2(Div(1, 3), x + 2), x)
    assert second == Mul2(Mul2(Div(1, 3), x), x + 2)
    assert first != second
    assert touch(first) == x*(x/3 + S(2)/3)
    assert touch(second) == x*(x + 2)/3


def test_structural_parenthesized_operands():
    x, y, z = symbols('x y z')
    assert parse_structural_expr('3*(x+1)') == Mul2(3, x + 1)
    assert parse_structural_expr('x*(x+2)/3') == Div(Mul2(x, x + 2), 3)
    assert parse_structural_expr('(x*y)*z') == Mul2(Mul2(x, y), z)
    assert parse_structural_expr('x*(y*z)') == Mul2(x, Mul2(y, z))
    assert touch(parse_structural_expr('x*(x+2)/3')) == x*(x + 2)/3


def test_structural_implicit_multiplication():
    x = symbols('x')
    assert parse_structural_expr(
        '2x', transformations='all') == Mul2(2, x)
