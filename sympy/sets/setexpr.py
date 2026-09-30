from __future__ import annotations
from sympy.core import Expr
from sympy.core.decorators import call_highest_priority, _sympifyit
from sympy.core.numbers import oo
from sympy.core.singleton import S
from .fancysets import ImageSet
from .sets import (FiniteSet, Interval, set_add, set_sub, set_mul, set_div,
    set_pow, set_function)


class SetExpr(Expr):
    """An expression that can take on values of a set.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet
    >>> from sympy.sets.setexpr import SetExpr

    >>> a = SetExpr(Interval(0, 5))
    >>> b = SetExpr(FiniteSet(1, 10))
    >>> (a + b).set
    Union(Interval(1, 6), Interval(10, 15))
    >>> (2*a + b).set
    Interval(1, 20)
    """
    _op_priority = 11.0

    def __new__(cls, setarg):
        return Expr.__new__(cls, setarg)

    set = property(lambda self: self.args[0])

    def _latex(self, printer):
        return r"SetExpr\left({}\right)".format(printer._print(self.set))

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return _setexpr_apply_operation(set_add, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return _setexpr_apply_operation(set_add, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return _setexpr_apply_operation(set_mul, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return _setexpr_apply_operation(set_mul, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return _setexpr_apply_operation(set_sub, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return _setexpr_apply_operation(set_sub, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return _setexpr_apply_operation(set_pow, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return _setexpr_apply_operation(set_pow, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return _setexpr_apply_operation(set_div, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return _setexpr_apply_operation(set_div, other, self)

    def _eval_func(self, func):
        # TODO: this could be implemented straight into `imageset`:
        res = set_function(func, self.set)
        if res is None:
            return SetExpr(ImageSet(func, self.set))
        return SetExpr(res)


def _setexpr_apply_operation(op, x, y):
    if isinstance(x, SetExpr):
        x = x.set
    if isinstance(y, SetExpr):
        y = y.set
    out = op(x, y)
    return SetExpr(out)


def _domain_from_assumptions(expr):
    """Return a conservative real value set implied by expr assumptions."""
    if not expr.is_Symbol or expr.is_real is not True:
        return None

    if expr.is_zero:
        return FiniteSet(0)

    if expr.is_positive:
        if expr.is_integer:
            if expr.is_prime:
                lo = 3 if expr.is_odd else 2
            elif expr.is_composite:
                lo = 9 if expr.is_odd else 4
            elif expr.is_even:
                lo = 2
            else:
                lo = 1
            return Interval(lo, oo)
        return Interval.open(0, oo)

    if expr.is_negative:
        if expr.is_integer:
            hi = -2 if expr.is_even else -1
            return Interval(-oo, hi)
        return Interval.open(-oo, 0)

    if expr.is_nonnegative:
        return Interval(0, oo)

    if expr.is_nonpositive:
        return Interval(-oo, 0)

    return Interval(-oo, oo)


def _setexpr_from_assumptions(expr):
    """Evaluate expr conservatively in the SetExpr domain."""
    if not expr.free_symbols:
        return expr

    if expr.is_Symbol:
        domain = _domain_from_assumptions(expr)
        return None if domain is None else SetExpr(domain)

    args = [_setexpr_from_assumptions(arg) for arg in expr.args]
    if any(arg is None for arg in args):
        return None

    if expr.is_Add:
        result = args[0]
        for arg in args[1:]:
            result += arg
        return result

    if expr.is_Mul:
        result = args[0]
        for arg in args[1:]:
            result *= arg
        return result

    if expr.is_Pow:
        base, exponent = args
        if isinstance(exponent, SetExpr) or exponent.is_integer is not True:
            return None
        if exponent.is_nonnegative:
            return base**exponent
        if isinstance(base, SetExpr) and base.set.contains(0) is S.false:
            return (1/base)**(-exponent)
        return None

    return None


def _value_set(expr):
    """Return a conservative set containing all values of expr.

    Only inexpensive SetExpr arithmetic is used. None is returned when
    the expression cannot be handled safely by the supported operations.
    """
    result = _setexpr_from_assumptions(expr)
    if result is None:
        return None
    if isinstance(result, SetExpr):
        return result.set
    return FiniteSet(result)
