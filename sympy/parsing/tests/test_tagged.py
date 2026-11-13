"""Regression tests for the experimental opt-in tagged arithmetic parser."""
from __future__ import annotations

from itertools import product
from random import Random

from sympy import Integral, Mul, S, Symbol, sin, sstr, symbols
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.tagged import annotation, parse_tagged_expr, tagged, touch


def _walk(e):
    yield e
    for arg in getattr(e, 'args', ()):
        yield from _walk(arg)


def _check(source, local_dict=None):
    expr = parse_tagged_expr(source, local_dict=local_dict)
    printed = sstr(expr, order='none')
    reparsed = parse_tagged_expr(printed, local_dict=local_dict)
    assert reparsed == expr, (source, printed, repr(reparsed), repr(expr))
    for e in _walk(expr):
        ann = annotation(e)
        if ann is not None:
            kind, _ = ann
            assert e.is_Mul and len(e.args) == 2
            assert e.args[0].assumptions0 == {
                'commutative': True, 'parse_' + kind: True,
            }
    raw = touch(expr)
    assert all(annotation(e) is None for e in _walk(raw))
    assert touch(raw) == raw  # idempotence

    got = touch(expr, evaluate=True)
    assert all(annotation(e) is None for e in _walk(got))
    expected = parse_expr(source, local_dict=local_dict)
    assert got == expected, (source, printed, got, expected)
    # Evaluating after detagging agrees with evaluating immediately.
    assert touch(raw, evaluate=True) == got

    return expr, printed


def test_canonical_nameless_symbol_tags():
    t1 = Symbol('', parse_division=True)
    t2 = Symbol('', parse_division=True)
    assert t1 is t2
    assert t1 != Symbol('', parse_power=True)
    expr = tagged('division', Symbol('x'))
    assert expr.args == (t1, Symbol('x'))
    assert len(expr.args) == 2
    assert annotation(expr) == ('division', Symbol('x'))
    assert t1 in expr.free_symbols  # Invisible in text, NOT free-symbol invisible.
    assert annotation(Mul(Symbol(''), Symbol('x'), evaluate=False)) is None


def test_operator_differences():
    assert _check('a/b')[1] == 'a/b'
    assert _check('a*b**-1')[1] == 'a*b**(-1)'
    assert _check('a+(-1*b)')[1] == 'a+(-1*b)'
    assert _check('a-b')[1] == 'a-b'
    assert _check('a*x**(-2*y)')[1] == 'a*x**(-2*y)'
    assert _check('-1**2')[1] == '-1**2'
    assert _check('(-1)**2')[1] == '(-1)**2'
    assert _check('(a/b)/(c/d)')[1] == '(a/b)/(c/d)'
    assert _check('x/(y**-1)')[1] == 'x/y**(-1)'
    assert _check('-(-(0))')[1] == '-(+0)'


def test_selected_regressions():
    cases = (
        'a/b', 'a*b**-1', 'a*x**(-2*y)', 'a*x**(-(2*y))',
        'a*x**(-2/y)', 'a/(b/c)', '(a/b)*c', 'a*(b/c)',
        '1/2', '2/4', '1/1', '-2*3/-1/2',
        'a-b', 'a+(-1*b)', '-1**2', '(-1)**2',
        '-(1/2)', 'x**(1/2)', 'x**-2', '2**-2',
        '-(a/b)', '-(-a)', '-(-1**2)',
        'a+(-b)', 'a+(-1*b)', 'a-(-b)', 'a-(-1*b)',
        '-a**b', '(-a)**b', '-a/b', '(-a)/b',
        'a/(-b)', 'a*(-b)', '(-a)*b',
        'a**(-b)', 'a**(-2*b)', 'a**(-b*c)',
        'a**(-(b*c))', 'a**(-b/c)', 'a**(-(b/c))',
        'a**(b-c)', 'a**(b+(-c))',
        '1/(a/b)', 'a/(-1)', 'a/(b-c)',
        'a**(b**(-1))', '(a**(-1))**b',
        'sin(a/b)', 'Integral(a/b,(a,0,1))',
        'a*(b+c)', 'a+(b*c)', 'a-(b*c)',
        'a+(-2*b)', 'a-(-2*b)',
        '3*a/(-2*a)', 'a**(1/2)', 'a**(2/3)',
        'a+b+c', 'a*b*c', 'a**b**c',
        'a/(b/c/d)', 'a-b-c', 'a-b+c',
        '-(-(0))', '-(-(-0))',
    )
    for case in cases:
        _check(case)


def test_ternary_arithmetic_corpus():
    ops = ('+', '-', '*', '/', '**')
    for a, b, c in product('abc', repeat=3):
        for first, second in product(ops, repeat=2):
            _check(f'({a}{first}{b}){second}{c}')
            _check(f'{a}{first}({b}{second}{c})')


def test_nested_fuzz_roundtrip():
    rng = Random(40711)
    ops = ('+', '-', '*', '/', '**')
    leaves = ('a', 'b', 'c', 'x', 'y', '1', '2', '-1', '-2', '0', '3/2')

    def generate(depth):
        if depth == 0 or (depth >= 1 and rng.randrange(5) == 0):
            return rng.choice(leaves)
        if rng.randrange(6) == 0:
            return '-(' + generate(depth - 1) + ')'
        left, right = generate(depth - 1), generate(depth - 1)
        return '(' + left + rng.choice(ops) + right + ')'

    # Check structure here; very deep numerical powers can grow explosively
    # when touched, so evaluation is handled in the smaller corpus above.
    for _ in range(3500):
        source = generate(rng.choice((2, 3, 4)))
        expr = parse_tagged_expr(source)
        printed = sstr(expr, order='none')
        assert parse_tagged_expr(printed) == expr, (source, printed)


def test_noncommutative_and_function_arguments():
    A, B, C = symbols('A B C', commutative=False)
    scope = {'A': A, 'B': B, 'C': C}
    for source in ('A*B', 'A/B', 'A*(B*C)', '(A*B)*C', 'A-(B/C)',
                   'A*B**-1', '(A/B)*C', 'A*(B/C)', 'A**(-2*B)'):
        _check(source, scope)

    for source in ('sin(a/b)', 'Integral(a/b, (a, 0, 1))',
                   'sqrt(a/b)', 'Eq(a/b, a*b)', 'sin(-1**2)'):
        _check(source)


def test_touch_does_not_call_doit():
    x = Symbol('x')
    e = parse_tagged_expr('Integral(1/x, (x, 1, 2))')

    raw = touch(e)
    assert raw.func == Integral
    assert raw.function == Mul(1, 1/x, evaluate=False)
    assert raw.function.args == (S.One, 1/x)

    assert touch(e, evaluate=True) == Integral(1/x, (x, 1, 2))


def test_selective_annotations():
    """Simple additions/products need no tag; ambiguous ones still do."""
    from sympy import Add, Pow

    assert annotation(parse_tagged_expr('a+b')) is None
    assert annotation(parse_tagged_expr('a*b')) is None
    assert annotation(parse_tagged_expr('(a+b)+c')) is None
    assert annotation(parse_tagged_expr('a*(b*c)')) is None

    assert annotation(parse_tagged_expr('a+(-b)'))[0] == 'addition'
    assert annotation(parse_tagged_expr('-2*a'))[0] == 'multiplication'
    assert annotation(parse_tagged_expr('a*b**-1'))[0] == 'multiplication'

    for source in (
        '(a+b)+c', 'a+(b+c)', 'a*(b*c)', '(a*b)*c',
        'a+(-b)', 'a+(-1*b)', '-2*a', '-2*(a+b)',
        'a*b**-1', 'a*x**(-2*y)', 'a/(b/c)', '(a/b)*c',
    ):
        _check(source)
