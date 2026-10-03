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
from sympy.core.exprtools import _decompose_exprs
from sympy.functions.combinatorial.factorials import binomial
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.recurrences import DifferentialRecurrence


def _is_zero(coeff):
    """Return True only when a coefficient is known to be zero."""
    return coeff == 0 or coeff.is_zero is True


def _polynomial_coeffs(expr, gen):
    """Return sparse coefficients of a structural polynomial in ``gen``.

    The result maps powers of ``gen`` to coefficients.  ``_decompose_exprs``
    analyzes the expression as written, so this does not expand products or
    otherwise invoke the higher-level polynomial conversion machinery.
    ``None`` is returned when a generator-dependent factor is not a power of
    ``gen`` with a nonnegative integer exponent.
    """
    (terms,), _ = _decompose_exprs(
        (expr,), is_coeff=lambda factor: not factor.has(gen))

    coeffs = {}
    for coeff_factors, powers in terms:
        degree = 0
        for base, (pos, neg) in powers.items():
            if base != gen or neg:
                return None
            degree += int(pos)

        coeff = S.One
        for factor in coeff_factors:
            coeff *= factor

        coeff = coeffs.get(degree, S.Zero) + coeff
        if _is_zero(coeff):
            coeffs.pop(degree, None)
        else:
            coeffs[degree] = coeff

    return coeffs


def _polynomial_power_data(expr, gen):
    """Return reversed polynomial data for ``base**n`` when structural."""
    exponent = 1
    base = expr
    if expr.is_Pow:
        if not (expr.exp.is_Integer and expr.exp.is_nonnegative):
            return None
        exponent = int(expr.exp)
        if exponent == 0:
            return {0: S.One}, 0, 0
        base = expr.base

    coeffs = _polynomial_coeffs(base, gen)
    if not coeffs:
        return None

    degree = max(coeffs)
    if degree == 0:
        return None

    # If B(x) = sum(c_k*x**k) has degree d, then
    # B(x) = x**d*A(1/x), where A(t) has coefficients c_{d-k}.
    reversed_coeffs = {
        degree - power: coeff for power, coeff in coeffs.items()
    }
    return reversed_coeffs, degree, exponent


def _recurrence_safe(data):
    """Return whether polynomial-power data is safe for exact recurrence."""
    coeffs, _, exponent = data
    if exponent == 0:
        return True

    leading = coeffs.get(0, S.Zero)
    if leading.is_zero is not False:
        return False

    # Repeated exact division can cause severe expression swell when the
    # polynomial coefficients contain symbolic parameters.  Keep the
    # recurrence shortcut numeric for now; specialized streams may still
    # handle symbolic data directly.
    return not any(coeff.free_symbols for coeff in coeffs.values())


def _poly_mul(left, right):
    """Multiply sparse ascending coefficient dictionaries."""
    if not left or not right:
        return {}

    result = {}
    for i, a in left.items():
        for j, b in right.items():
            power = i + j
            coeff = result.get(power, S.Zero) + a*b
            if _is_zero(coeff):
                result.pop(power, None)
            else:
                result[power] = coeff
    return result


def _poly_diff(coeffs):
    """Differentiate sparse ascending coefficient dictionaries."""
    result = {}
    for power, coeff in coeffs.items():
        if power:
            value = power*coeff
            if not _is_zero(value):
                result[power - 1] = value
    return result


def _polynomial_product_annihilator(factors):
    r"""Return ``(P, Q)`` for ``Q*F' - P*F = 0``.

    Each item in *factors* is ``(A, degree, m)``, where ``A`` is the sparse
    ascending coefficient dictionary for a reversed polynomial factor and

    ``F(t) = product(A_i(t)**m_i)``.

    If ``P/Q`` is the logarithmic derivative for the factors processed so
    far, adjoining ``A(t)**m`` updates

    ``Q -> Q*A`` and
    ``P -> P*A + m*A'*Q``.
    """
    p = {}
    q = {0: S.One}

    for coeffs, _, exponent in factors:
        if exponent == 0:
            continue

        old_q = q
        q = _poly_mul(q, coeffs)
        p = _poly_mul(p, coeffs)

        extra = _poly_mul(old_q, _poly_diff(coeffs))
        for power, coeff in extra.items():
            value = p.get(power, S.Zero) + exponent*coeff
            if _is_zero(value):
                p.pop(power, None)
            else:
                p[power] = value

    return p, q


class _StreamStats:
    """Optional counters for studying the amount of lazy stream work."""

    __slots__ = (
        'term_requests', 'child_requests', 'cache_hits', 'cache_misses',
        'generated_terms', 'mul_states_popped', 'cancelled_layers',
        'streams_created',
    )

    def __init__(self):
        self.term_requests = 0
        self.child_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.generated_terms = 0
        self.mul_states_popped = 0
        self.cancelled_layers = 0
        self.streams_created = 0

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


class _TermStream:
    """Memoized stream of descending ``(degree, coefficient)`` pairs."""

    def __init__(self, stats=None):
        self._stats = stats
        if stats is not None:
            stats.streams_created += 1
        self._cache = []
        self._done = False
        self._generator = self._generate()

    def _generate(self):
        raise NotImplementedError

    def term(self, index):
        """Return term *index*, generating only as much as necessary."""
        stats = self._stats
        if stats is not None:
            stats.term_requests += 1
            if index < len(self._cache):
                stats.cache_hits += 1
            else:
                stats.cache_misses += 1

        while len(self._cache) <= index and not self._done:
            try:
                self._cache.append(next(self._generator))
                if stats is not None:
                    stats.generated_terms += 1
            except StopIteration:
                self._done = True
        if index < len(self._cache):
            return self._cache[index]
        return None

    def _child_term(self, child, index):
        if self._stats is not None:
            self._stats.child_requests += 1
        return child.term(index)

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

    def __init__(self, degree, coeff, stats=None):
        self.degree = degree
        self.coeff = coeff
        super().__init__(stats)

    def _generate(self):
        if not _is_zero(self.coeff):
            yield self.degree, self.coeff


class _ScaleStream(_TermStream):
    """Multiply all coefficients of a child stream by a scalar."""

    def __init__(self, factor, child, stats=None):
        self.factor = factor
        self.child = child
        super().__init__(stats)

    def _generate(self):
        if _is_zero(self.factor):
            return

        index = 0
        while True:
            term = self._child_term(self.child, index)
            if term is None:
                return
            degree, coeff = term
            coeff *= self.factor
            if not _is_zero(coeff):
                yield degree, coeff
            index += 1


class _AffinePowerStream(_TermStream):
    """Direct coefficient stream for ``(a*x + b)**n``.

    With ``t = 1/x``, this is the coefficient sequence of
    ``a**n * (1 + (b/a)*t)**n``.  Emitting it directly avoids constructing
    the multiplication lattice that exponentiation by squaring would create.
    """

    def __init__(self, a, b, exponent, stats=None):
        self.a = a
        self.b = b
        self.exponent = exponent
        super().__init__(stats)

    def _generate(self):
        n = self.exponent

        if _is_zero(self.a):
            coeff = self.b**n
            if not _is_zero(coeff):
                yield 0, coeff
            return

        if _is_zero(self.b):
            coeff = self.a**n
            if not _is_zero(coeff):
                yield n, coeff
            return

        for r in range(n + 1):
            coeff = binomial(n, r)*self.a**(n - r)*self.b**r
            if not _is_zero(coeff):
                yield n - r, coeff


class _PolynomialPowersStream(_TermStream):
    r"""Direct stream for a product of polynomial powers.

    With ``t = 1/x`` the coefficient generator is

    ``F(t) = product(A_i(t)**m_i)``,

    where the ``A_i`` are reversed polynomial bases.  Its logarithmic
    derivative gives a first-order differential annihilator
    ``Q(t)*F'(t) - P(t)*F(t)``.  ``DifferentialRecurrence`` converts that
    annihilator to a finite-order coefficient recurrence, so each new degree
    layer is obtained without walking a multidimensional convolution.
    """

    def __init__(self, factors, stats=None):
        self.factors = factors
        super().__init__(stats)

    def _generate(self):
        factors = [factor for factor in self.factors if factor[2] != 0]
        total = sum(
            degree*exponent for _, degree, exponent in factors)

        p, q = _polynomial_product_annihilator(factors)
        recurrence_terms = []
        recurrence_terms.extend(
            (0, power, -coeff)
            for power, coeff in p.items()
            if not _is_zero(coeff)
        )
        recurrence_terms.extend(
            (1, power, coeff)
            for power, coeff in q.items()
            if not _is_zero(coeff)
        )
        recurrence = DifferentialRecurrence(recurrence_terms)

        coeff = S.One
        for poly, _, exponent in factors:
            coeff *= poly[0]**exponent

        coefficients = [coeff]
        if not _is_zero(coeff):
            yield total, coeff

        history_shifts = [
            shift for shift in recurrence.shifts if shift != 1
        ]
        for r in range(total):
            numerator = S.Zero
            for shift in history_shifts:
                index = r + shift
                if index >= 0:
                    numerator += (
                        recurrence.coefficient(shift, r)*coefficients[index]
                    )

            next_coeff = -numerator/recurrence.coefficient(1, r)
            coefficients.append(next_coeff)
            if not _is_zero(next_coeff):
                yield total - r - 1, next_coeff


class _AddStream(_TermStream):
    """Merge descending child streams, combining equal degree layers."""

    def __init__(self, children, stats=None):
        self.children = children
        super().__init__(stats)

    def _generate(self):
        heap = []
        for child_index, child in enumerate(self.children):
            term = self._child_term(child, 0)
            if term is not None:
                degree, coeff = term
                heappush(heap, (-degree, child_index, 0, coeff))

        while heap:
            degree = -heap[0][0]
            coeff = S.Zero

            while heap and -heap[0][0] == degree:
                _, child_index, term_index, child_coeff = heappop(heap)
                coeff += child_coeff

                term = self._child_term(
                    self.children[child_index], term_index + 1)
                if term is not None:
                    next_degree, next_coeff = term
                    heappush(
                        heap,
                        (-next_degree, child_index, term_index + 1,
                         next_coeff),
                    )

            if _is_zero(coeff):
                if self._stats is not None:
                    self._stats.cancelled_layers += 1
            else:
                yield degree, coeff


class _MulStream(_TermStream):
    """Lazily merge the ordered Cartesian product of two term streams."""

    def __init__(self, left, right, stats=None):
        self.left = left
        self.right = right
        super().__init__(stats)

    def _state(self, left_index, right_index):
        left = self._child_term(self.left, left_index)
        right = self._child_term(self.right, right_index)
        if left is None or right is None:
            return None
        return left[0] + right[0], left[1] * right[1]

    def _generate(self):
        first = self._state(0, 0)
        if first is None:
            return

        heap = [(-first[0], 0, 0, first[1])]
        seen = {(0, 0)}

        while heap:
            degree = -heap[0][0]
            coeff = S.Zero

            while heap and -heap[0][0] == degree:
                _, left_index, right_index, state_coeff = heappop(heap)
                if self._stats is not None:
                    self._stats.mul_states_popped += 1
                coeff += state_coeff

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
                        heappush(heap, (-state[0], *neighbor, state[1]))

            if _is_zero(coeff):
                if self._stats is not None:
                    self._stats.cancelled_layers += 1
            else:
                yield degree, coeff


def _power_stream(base, exponent, stats=None):
    """Build a logarithmic-depth lazy stream for ``base**exponent``."""
    if exponent == 0:
        return _MonomialStream(0, S.One, stats)
    if exponent == 1:
        return base

    half = _power_stream(base, exponent // 2, stats)
    square = _MulStream(half, half, stats)
    if exponent % 2:
        return _MulStream(square, base, stats)
    return square


def _stream_from_expr(expr, gen, stats=None):
    """Build the most specific lazy stream available for ``expr``.

    Multiplication first looks for a favorable structure that can be emitted
    directly, such as a recurrence-backed product of polynomial powers::

                         Mul
                          |
                 can I recognize a
              favorable special structure?
                    /             \
                  yes              no
                   |                |
            recurrence stream    generic MulStream
                   |                |
              C0, C1, C2, ...   walk diagonals

    Falling back to ``_MulStream`` does not expand the product.  It lazily
    traverses the ordered Cartesian product of the child term streams, one
    degree diagonal at a time, and stops as soon as the consumer has enough
    output.
    """
    if expr == gen:
        return _MonomialStream(1, S.One, stats)

    if not expr.has(gen):
        return _MonomialStream(0, expr, stats)

    if expr.is_Add:
        return _AddStream(
            [_stream_from_expr(arg, gen, stats) for arg in expr.args], stats)

    if expr.is_Mul:
        scale = S.One
        dependent = []
        for arg in expr.args:
            if arg.has(gen):
                dependent.append(arg)
            else:
                scale *= arg

        if len(dependent) >= 2:
            polynomial_factors = [
                _polynomial_power_data(arg, gen) for arg in dependent
            ]
            if all(
                    factor is not None and _recurrence_safe(factor)
                    for factor in polynomial_factors):
                stream = _PolynomialPowersStream(polynomial_factors, stats)
                if scale != 1:
                    stream = _ScaleStream(scale, stream, stats)
                return stream

        children = [_stream_from_expr(arg, gen, stats) for arg in dependent]
        stream = children[0]
        for child in children[1:]:
            stream = _MulStream(stream, child, stats)
        if scale != 1:
            stream = _ScaleStream(scale, stream, stats)
        return stream

    if expr.is_Pow:
        if expr.exp.is_Integer and expr.exp.is_nonnegative:
            exponent = int(expr.exp)
            if exponent == 0:
                return _MonomialStream(0, S.One, stats)

            data = _polynomial_power_data(expr, gen)
            if data is not None:
                coeffs, degree, exponent = data
                if degree == 1:
                    return _AffinePowerStream(
                        coeffs.get(0, S.Zero),
                        coeffs.get(1, S.Zero),
                        exponent,
                        stats,
                    )
                if _recurrence_safe(data):
                    return _PolynomialPowersStream([data], stats)

            base = _stream_from_expr(expr.base, gen, stats)
            return _power_stream(base, exponent, stats)

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

    Set ``collect_stats=True`` to collect experimental counters describing
    how much lazy work was needed.  The current values are available from the
    ``stats`` property.

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

    def __init__(self, expr, gen, collect_stats=False):
        expr = sympify(expr)
        gen = sympify(gen)
        if not gen.is_Symbol:
            raise TypeError("the prototype requires a Symbol generator")
        self.expr = expr
        self.gen = gen
        self._stats = _StreamStats() if collect_stats else None
        self._stream = _stream_from_expr(expr, gen, self._stats)

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

    @property
    def stats(self):
        """Return instrumentation counters, or None when not collecting."""
        if self._stats is None:
            return None
        result = self._stats.as_dict()
        result['root_terms_generated'] = len(self._stream._cache)
        return result
