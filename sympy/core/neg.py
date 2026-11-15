"""Experimental subtraction marker for gh-4596.

Binary subtraction retains a Neg node until its operand becomes an Add.
This is a prototype, not a new public API.
"""
from .function import Function
from .singleton import S
from .add import Add


class Neg(Function):
    """An explicit negative term that can distribute over an Add."""

    nargs = 1

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            return -arg
        if arg.is_Mul and arg.as_coeff_Mul()[0] is not S.One:
            # Ordinary numeric multiplication remains undistributed.
            return -arg
        if isinstance(arg, cls):
            return arg.args[0]
        if arg.is_Add:
            return Add(*(cls(t) for t in arg.args))

    def __neg__(self):
        return self.args[0]

    def could_extract_minus_sign(self):
        return True

    def _eval_is_commutative(self):
        return self.args[0].is_commutative

    def _eval_is_finite(self):
        return self.args[0].is_finite

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_real(self):
        return self.args[0].is_real

    def _eval_is_rational(self):
        return self.args[0].is_rational

    def _eval_is_integer(self):
        return self.args[0].is_integer

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_positive(self):
        return self.args[0].is_negative

    def _eval_is_negative(self):
        return self.args[0].is_positive

    def _eval_is_nonnegative(self):
        return self.args[0].is_nonpositive

    def _eval_is_nonpositive(self):
        return self.args[0].is_nonnegative

    def _eval_derivative(self, symbol):
        return -self.args[0].diff(symbol)
