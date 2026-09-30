from __future__ import annotations
from sympy.core import Basic, Expr
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.multipledispatch import Dispatcher
from sympy.sets.setexpr import set_mul
from sympy.sets.sets import Interval, Set


_x, _y = symbols("x y")


_set_mul = Dispatcher('_set_mul')
_set_div = Dispatcher('_set_div')


@_set_mul.register(Basic, Basic)
def _(x, y):
    return None

@_set_mul.register(Set, Set)
def _(x, y):
    return None

@_set_mul.register(Expr, Expr)
def _(x, y):
    return x*y

def _interval_mul_endpoint(x, y):
    # An infinite endpoint is not an element of an Interval. When paired
    # with a zero endpoint, the limiting product is 0 rather than the
    # indeterminate scalar product 0*oo.
    if x.is_zero is True or y.is_zero is True:
        return S.Zero
    return x*y


@_set_mul.register(Interval, Interval)
def _(x, y):
    """
    Multiplications in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    comvals = (
        (_interval_mul_endpoint(x.start, y.start),
            bool(x.left_open or y.left_open)),
        (_interval_mul_endpoint(x.start, y.end),
            bool(x.left_open or y.right_open)),
        (_interval_mul_endpoint(x.end, y.start),
            bool(x.right_open or y.left_open)),
        (_interval_mul_endpoint(x.end, y.end),
            bool(x.right_open or y.right_open)),
    )
    # TODO: handle symbolic intervals
    minval = min(v for v, _ in comvals)
    maxval = max(v for v, _ in comvals)
    # An extremum is closed if any corner attaining it is closed.
    minopen = all(o for v, o in comvals if v == minval)
    maxopen = all(o for v, o in comvals if v == maxval)
    return Interval(
        minval,
        maxval,
        minopen,
        maxopen
    )

@_set_div.register(Basic, Basic)
def _(x, y):
    return None

@_set_div.register(Expr, Expr)
def _(x, y):
    return x/y

@_set_div.register(Set, Set)
def _(x, y):
    return None

@_set_div.register(Interval, Interval)
def _(x, y):
    """
    Divisions in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    if (y.start*y.end).is_negative:
        return Interval(-oo, oo)
    if y.start == 0:
        s2 = oo
    else:
        s2 = 1/y.start
    if y.end == 0:
        s1 = -oo
    else:
        s1 = 1/y.end
    return set_mul(x, Interval(s1, s2, y.right_open, y.left_open))
