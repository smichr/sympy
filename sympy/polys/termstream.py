"""Experimental lazy term streams for univariate polynomial expressions.

This module is a prototype.  It represents an unexpanded polynomial
expression as a stream of ``(degree, coefficient)`` pairs in descending
degree order.  Addition merges child streams and multiplication lazily
enumerates the ordered Cartesian product of two child streams.

Only expressions built from a single Symbol generator, coefficients
independent of that generator, Add, Mul, and Pow with a nonnegative
integer exponent are supported.
"""

from heapq import heappop, heappush

from sympy.core import S, sympify
from sympy.polys.polyerrors import PolynomialError


def _is_zero(coeff):
    """Return True only when a coefficient is known to be zero."""
    return coeff == 0 or coeff.is_zero is True


class _TermStream:
    """Memoized stream of descending ``(degree, coefficient)`` pairs."""

    def __init__(self):
        self._cache = []
        self._done = False
        self._generator = self._generate()

    def _generate(self):
        raise NotImplementedError

    def term(self, index):
        """Return term *index*, generating only as much as necessary."""
        while len(self._cache) <= index and not self._done:
            try:
                self._cache.append(next(self._generator))
            except StopIteration:
                self._done = True
        if index < len(self._cache):
            return self._cache[index]
        return None

    def take(self, count):
        """Return at most the first *count* emitted terms."""
        result = []
        for index in range(count):
            term = self.term(index)
            if term is None:
                break
            result.append(term)
        return result


class _MonomialStream(_TermStream):

    def __init__(self, degree, coeff):
        self.degree = degree
        self.coeff = coeff
        super().__init__()

    def _generate(self):
        if not _is_zero(self.coeff):
            yield self.degree, self.coeff


class _AddStream(_TermStream):
    """Merge descending child streams, combining equal degree layers."""

    def __init__(self, children):
        self.children = children
        super().__init__()

    def _generate(self):
        heap = []
        for child_index, child in enumerate(self.children):
            term = child.term(0)
            if term is not None:
                degree, coeff = term
                heappush(heap, (-degree, child_index, 0, coeff))

        while heap:
            degree = -heap[0][0]
            coeff = S.Zero

            while heap and -heap[0][0] == degree:
                _, child_index, term_index, child_coeff = heappop(heap)
                coeff += child_coeff

                term = self.children[child_index].term(term_index + 1)
                if term is not None:
                    next_degree, next_coeff = term
                    heappush(
                        heap,
                        (-next_degree, child_index, term_index + 1,
                         next_coeff),
                    )

            if not _is_zero(coeff):
                yield degree, coeff


class _MulStream(_TermStream):
    """Lazily merge the ordered Cartesian product of two term streams."""

    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def _state(self, left_index, right_index):
        left = self.left.term(left_index)
        right = self.right.term(right_index)
        if left is None or right is None:
            return None
        return left[0] + right[0], left[1] * right[1]

    def _generate(self):
        first = self._state(0, 0)
        if first is None:
            return

        heap = [(-first[0], 0, 0)]
        seen = {(0, 0)}

        while heap:
            degree = -heap[0][0]
            coeff = S.Zero

            # Processing a state can reveal another state having the same
            # degree, so continue until the heap really moves below this
            # degree layer.
            while heap and -heap[0][0] == degree:
                _, left_index, right_index = heappop(heap)
                state = self._state(left_index, right_index)
                coeff += state[1]

                neighbors = (
                    (left_index + 1, right_index),
                    (left_index, right_index + 1),
                )
                for neighbor in neighbors:
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    state = self._state(*neighbor)
                    if state is not None:
                        heappush(heap, (-state[0], *neighbor))

            if not _is_zero(coeff):
                yield degree, coeff


def _power_stream(base, exponent):
    """Build a logarithmic-depth lazy stream for ``base**exponent``."""
    if exponent == 0:
        return _MonomialStream(0, S.One)
    if exponent == 1:
        return base

    half = _power_stream(base, exponent // 2)
    square = _MulStream(half, half)
    if exponent % 2:
        return _MulStream(square, base)
    return square


def _stream_from_expr(expr, gen):
    if expr == gen:
        return _MonomialStream(1, S.One)

    if not expr.has(gen):
        return _MonomialStream(0, expr)

    if expr.is_Add:
        return _AddStream([_stream_from_expr(arg, gen) for arg in expr.args])

    if expr.is_Mul:
        children = [_stream_from_expr(arg, gen) for arg in expr.args]
        stream = children[0]
        for child in children[1:]:
            stream = _MulStream(stream, child)
        return stream

    if expr.is_Pow:
        if expr.exp.is_Integer and expr.exp.is_nonnegative:
            base = _stream_from_expr(expr.base, gen)
            return _power_stream(base, int(expr.exp))

    raise PolynomialError(
        "%s is not supported as a polynomial expression in %s"
        % (expr, gen)
    )


class PolyTermStream:
    """Lazy descending term stream for a univariate polynomial expression.

    This is an experimental helper for studying polynomial queries that do
    not require complete expansion.  Coefficients are expressions independent
    of ``gen``.  A coefficient whose zero status is unknown is treated as
    nonzero; the current prototype is intended primarily for exact domains
    such as ``ZZ`` and ``QQ``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.polys.termstream import PolyTermStream
    >>> x = symbols('x')
    >>> PolyTermStream((x + 1)**100000, x).take(2)
    [(100000, 1), (99999, 100000)]
    >>> PolyTermStream((x + 1)**100000 - (x - 1)**100000, x).degree()
    99999
    """

    def __init__(self, expr, gen):
        expr = sympify(expr)
        gen = sympify(gen)
        if not gen.is_Symbol:
            raise TypeError("the prototype requires a Symbol generator")
        self.expr = expr
        self.gen = gen
        self._stream = _stream_from_expr(expr, gen)

    def term(self, index):
        """Return the *index*-th surviving term as ``(degree, coeff)``."""
        return self._stream.term(index)

    def take(self, count):
        """Return at most the first *count* surviving terms."""
        return self._stream.take(count)

    def degree(self):
        """Return the degree, or negative infinity for the zero polynomial."""
        term = self.term(0)
        if term is None:
            return S.NegativeInfinity
        return term[0]

    def LC(self):
        """Return the leading coefficient, or zero for the zero polynomial."""
        term = self.term(0)
        if term is None:
            return S.Zero
        return term[1]
