"""Low-level helpers for coefficient recurrences."""

from sympy.core import S


class DifferentialRecurrence:
    r"""Coefficient recurrence induced by a differential annihilator.

    ``terms`` is an iterable of triples ``(i, k, c)`` representing the
    monomials ``c*x**k*D**i`` of a differential operator.  If the operator
    annihilates

    .. math::

        F(x) = \sum_{n \geq 0} a_n x^n,

    then each monomial contributes

    .. math::

        c (n-k+1)^{\overline{i}} a_{n+i-k}

    to the coefficient recurrence.  The sequence shift is therefore
    ``i - k``.

    This object deliberately knows nothing about ``Poly`` or holonomic
    functions.  It only records this local conversion, so callers can either
    ask for the symbolic recurrence terms or evaluate a shift coefficient at
    a particular ``n``.
    """

    def __init__(self, terms):
        grouped = {}
        for derivative_order, power, coeff in terms:
            if coeff == 0:
                continue
            key = derivative_order - power, power
            grouped[key] = grouped.get(key, S.Zero) + coeff

        self._terms = tuple(
            (shift, power, shift + power, coeff)
            for (shift, power), coeff in sorted(grouped.items())
            if coeff != 0
        )
        self.shifts = tuple(sorted({term[0] for term in self._terms}))

    @staticmethod
    def _term_value(coeff, power, derivative_order, n):
        value = coeff
        for j in range(derivative_order):
            value *= n - power + 1 + j
        return value

    def terms(self, n):
        """Return ``{(shift, power): coefficient}`` at sequence index ``n``."""
        return {
            (shift, power): self._term_value(
                coeff, power, derivative_order, n)
            for shift, power, derivative_order, coeff in self._terms
        }

    def coefficient(self, shift, n):
        """Return the total coefficient multiplying ``a[n + shift]``."""
        result = S.Zero
        for term_shift, power, derivative_order, coeff in self._terms:
            if term_shift == shift:
                result += self._term_value(
                    coeff, power, derivative_order, n)
        return result
