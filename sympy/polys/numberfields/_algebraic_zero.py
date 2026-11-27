"""Experimental exact zero tests for expressions with rational radicals."""
from __future__ import annotations

from functools import cache
from itertools import product
from math import ceil, floor, lcm

from sympy.core.add import Add
from sympy.core.intfunc import integer_nthroot
from sympy.core.numbers import I, Rational
from sympy.core.singleton import S
from sympy.polys.domains import GF, QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.monomials import MonomialOps
from sympy.polys.sparsetools import smp_add


class _Unsupported(Exception):
    """The expression is outside the supported class or budget."""


# Bound the radical tower dimension, coefficient sizes and scalar work.
_MAX_POWER = 12
_MAX_NODES = 128
_MAX_DEGREE = 256
_MAX_WORK = 200_000
_MAX_BITS = 4096


def _bounded_rational(value):
    return max(abs(int(value.p)).bit_length(), int(value.q).bit_length()) <= _MAX_BITS


def _check_coefficient(value):
    if max(abs(int(value.numerator)).bit_length(),
            int(value.denominator).bit_length()) > _MAX_BITS:
        raise _Unsupported


def _radical_denominators(expr):
    """Recognize supported syntax and collect common root denominators."""
    pending = [expr]
    seen = set()
    denominators = {}
    degree = 1
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if len(seen) > _MAX_NODES or len(node.args) > _MAX_NODES:
            raise _Unsupported
        if node.is_Rational:
            if not _bounded_rational(node):
                raise _Unsupported
        elif node is I:
            pass
        elif node.is_Add or node.is_Mul:
            pending.extend(node.args)
        elif node.is_Pow and node.exp.is_Rational:
            p, q = map(int, node.exp.as_numer_denom())
            if max(abs(p), q) > _MAX_POWER:
                raise _Unsupported
            pending.append(node.base)
            if q != 1:
                old = denominators.get(node.base, 1)
                common = lcm(old, q)
                degree = degree // old * common
                if common > _MAX_POWER or degree > _MAX_DEGREE:
                    raise _Unsupported
                denominators[node.base] = common
        else:
            raise _Unsupported
    if I in seen and degree * 2 > _MAX_DEGREE:
        raise _Unsupported
    return denominators


def _real_enclosure(expr, bits):
    """Enclose supported real expressions in outward-rounded dyadic bounds.

    No numerical assumptions are used. If a reciprocal might have a zero
    denominator, or a fractional power might have a negative base, the
    result is unknown.
    """
    scale = 2**bits

    def out(lo, hi):
        if not (_bounded_rational(lo) and _bounded_rational(hi)):
            raise _Unsupported
        return Rational(floor(lo*scale), scale), Rational(ceil(hi*scale), scale)

    def int_power(lo, hi, p):
        if p < 0:
            if lo <= 0 <= hi:
                raise _Unsupported
            lo, hi = 1/hi, 1/lo
            p = -p
        a, b = lo**p, hi**p
        if p and p % 2 == 0 and lo < 0 < hi:
            return S.Zero, max(a, b)
        return min(a, b), max(a, b)

    @cache
    def scan(node):
        if node.is_Rational:
            return out(node, node)
        if node.is_Add:
            a, b = zip(*(scan(arg) for arg in node.args), strict=True)
            return out(sum(a), sum(b))
        if node.is_Mul:
            lo = hi = S.One
            for arg in node.args:
                a, b = scan(arg)
                products = [lo*a, lo*b, hi*a, hi*b]
                lo, hi = out(min(products), max(products))
            return lo, hi
        if node.is_Pow and node.exp.is_Rational:
            p, q = map(int, node.exp.as_numer_denom())
            if max(abs(p), q) > _MAX_POWER:
                raise _Unsupported
            lo, hi = scan(node.base)
            if q > 1:
                if lo < 0:
                    raise _Unsupported
                low = integer_nthroot(floor(lo*scale**q), q)[0]
                high = integer_nthroot(floor(hi*scale**q), q)[0]
                lo = Rational(low, scale)
                # The floor of the scaled root is exact; round the upper
                # bound up unless it is already exactly the root.
                high = Rational(high, scale)
                hi = high if high**q == hi else high + Rational(1, scale)
            return out(*int_power(lo, hi, p))
        raise _Unsupported

    try:
        return scan(expr)
    except _Unsupported:
        return None


class _RadicalTower:
    """Bounded arithmetic in a triangular monic rational quotient algebra.

    For generators a_i with a_i**q_i = f_i(a_0, ..., a_{i-1}), a basis
    consists of monomials having exponents strictly less than q_i. Every
    reduction by a relation terminates, without a general Groebner basis.
    """

    def __init__(self, n):
        self.n = n
        self.monomial_mul = MonomialOps(n).mul()
        self.constant = (0,)*n
        self.one = {self.constant: QQ.one}
        self.degrees = []
        self.relations = []
        self.dimension = 1
        self.remaining = _MAX_WORK
        self.reduced = {}

    def charge(self, n=1):
        self.remaining -= n
        if self.remaining < 0:
            raise _Unsupported

    def term(self, dest, monomial, coeff):
        self.charge()
        coeff += dest.get(monomial, QQ.zero)
        _check_coefficient(coeff)
        if coeff:
            dest[monomial] = coeff
        else:
            dest.pop(monomial, None)

    def add(self, a, b):
        self.charge(len(a) + len(b))
        out = smp_add(a, b, QQ, self.n)
        for m in b:
            _check_coefficient(out.get(m, QQ.zero))
        return out

    def root(self, a, q):
        i = len(self.degrees)
        self.dimension *= q
        if i >= self.n or self.dimension > _MAX_DEGREE:
            raise _Unsupported
        self.degrees.append(q)
        self.relations.append(a)
        m = [0]*self.n
        m[i] = 1
        return {tuple(m): QQ.one}

    def reduce(self, m):
        self.charge()
        if m in self.reduced:
            return self.reduced[m]
        for i in range(len(self.degrees) - 1, -1, -1):
            q = self.degrees[i]
            if m[i] >= q:
                v = list(m)
                v[i] -= q
                out = {}
                for k, c in self.relations[i].items():
                    new = self.monomial_mul(v, k)
                    for j, d in self.reduce(new).items():
                        self.term(out, j, c*d)
                self.reduced[m] = out
                return out
        return {m: QQ.one}

    def mul(self, a, b):
        if a == self.one:
            return b
        if b == self.one:
            return a
        out = {}
        for m, c in a.items():
            for n, d in b.items():
                self.charge()
                for k, e in self.reduce(self.monomial_mul(m, n)).items():
                    self.term(out, k, c*d*e)
        return out

    def power(self, a, p):
        out = self.one
        while p:
            if p & 1:
                out = self.mul(out, a)
            p //= 2
            if p:
                a = self.mul(a, a)
        return out

    def nonzero(self, a):
        """Can multiplication by a be certified invertible modulo a prime?

        Full rank modulo a prime implies full rank over QQ and therefore
        a cannot vanish at any solution of the defining relations. Failure
        to certify invertibility says nothing about the selected root.
        """
        if not a:
            return False
        if len(a) == 1 and self.constant in a:
            return True
        basis = [m + (0,)*(self.n - len(m)) for m in
            product(*(range(q) for q in self.degrees))]
        index = {m: i for i, m in enumerate(basis)}
        columns = [self.mul(a, {m: QQ.one}) for m in basis]
        n = self.dimension
        for prime in (65521, 1000003):
            if any(c.denominator % prime == 0
                    for col in columns for c in col.values()):
                continue
            field = GF(prime)
            rows = {}
            for i, col in enumerate(columns):
                row = {}
                for m, c in col.items():
                    self.charge()
                    r = int(c.numerator)*pow(int(c.denominator), -1, prime) % prime
                    if r:
                        row[index[m]] = field(r)
                if not row:
                    break
                rows[i] = row
            if len(rows) != n:
                continue
            matrix = DomainMatrix(rows, (n, n), field)
            # A block triangular matrix is invertible iff each diagonal
            # block is invertible. Bound sparse Gauss-Jordan work: after
            # k pivots, cancellation involves at most k rows of length
            # n-k+1. Forward cancellation may also occur for singular
            # rows, so use the worst case for every row in a block.
            components = matrix.scc()
            cost = sum(size**2 + size*((size + 1)**2//4)
                + sum(k*(size - k + 1) for k in range(size))
                for size in map(len, components))
            if cost > self.remaining:
                continue
            self.charge(cost)
            if all(len(matrix.extract(block, block).rref(method="GJ")[1])
                    == len(block) for block in components):
                return True
        return False


def _relation_zero(expr, *, only_zero=False):
    """Test an exact radical expression without choosing conjugates.

    Represent values as numerator/denominator pairs in a finite monic
    quotient algebra. For the root of A/B, introduce z satisfying
    z**q = A*B**(q - 1), so the actual root is represented by z/B.
    This avoids inverse generators and nonmonic relations.
    """
    denominators = _radical_denominators(expr)
    tower = _RadicalTower(len(denominators) + 1)
    one = tower.one

    @cache
    def root(base):
        q = denominators[base]
        a, b = scan(base)
        z = tower.root(tower.mul(a, tower.power(b, q - 1)), q)
        return z, b

    @cache
    def scan(node):
        if node.is_Rational:
            return ({tower.constant: QQ.from_sympy(node)}
                if node else {}), one
        if node is I:
            return tower.root({tower.constant: -QQ.one}, 2), one
        if node.is_Add:
            a, b = {}, one
            for arg in node.args:
                c, d = scan(arg)
                if b == d:
                    a = tower.add(a, c)
                else:
                    a = tower.add(tower.mul(a, d), tower.mul(c, b))
                    b = tower.mul(b, d)
            return a, b
        if node.is_Mul:
            a = b = one
            for arg in node.args:
                c, d = scan(arg)
                a, b = tower.mul(a, c), tower.mul(b, d)
            return a, b
        if node.is_Pow:
            p, q = map(int, node.exp.as_numer_denom())
            if q != 1:
                common = denominators[node.base]
                a, b = root(node.base)
                p *= common // q
            else:
                a, b = scan(node.base)
            if abs(p) > _MAX_POWER:
                raise _Unsupported
            if p < 0:
                bounds = _real_enclosure(node.base, 64)
                if bounds is None or bounds[0] <= 0 <= bounds[1]:
                    if not tower.nonzero(a):
                        raise _Unsupported
                a, b = b, a
                p = -p
            return tower.power(a, p), tower.power(b, p)
        raise _Unsupported

    a, b = scan(expr)
    if not b:
        return None
    if not a:
        return True
    if len(a) == 1 and tower.constant in a:
        return False
    if only_zero:
        return None
    if tower.nonzero(a):
        return False
    return None


def _node_count(expr):
    seen = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(node.args)
    return len(seen)


def _isolated_radical_zero(expr):
    if not expr.is_Add:
        return None
    radicals = []
    for term in expr.args:
        coeff, radical = term.as_coeff_Mul()
        if (coeff.is_Rational and coeff and radical.is_Pow
                and radical.exp.is_Rational and radical.exp.q > 1
                and radical.exp.p > 0):
            radicals.append((term, coeff, radical))
    if not radicals:
        return None
    # A cleared power identity needs a sign check to select the real root.
    term, coeff, radical = max(radicals, key=lambda r: _node_count(r[2]))
    rest = list(expr.args)
    rest.remove(term)
    rhs = -Add(*rest)/coeff
    base = _real_enclosure(radical.base, 64)
    rhs_bounds = _real_enclosure(rhs, 64)
    if (base is not None and base[0] >= 0
            and rhs_bounds is not None and rhs_bounds[0] >= 0):
        p, q = radical.exp.as_numer_denom()
        return _relation_zero(rhs**q - radical.base**p)
    return None


def _is_algebraic_zero(expr):
    """Return True, False, or None using bounded exact algebraic certificates.

    The supported class is finite expressions of rational constants, I,
    addition, multiplication and rational powers. Unsupported expressions
    and inconclusive tests return None. This does not compute a minimal
    polynomial or attempt to select a conjugate numerically.
    """
    try:
        _radical_denominators(expr)
        for bits in (24, 64, 128, 512):
            bounds = _real_enclosure(expr, bits)
            if bounds is None:
                break
            if bounds == (0, 0):
                return True
            if bounds[0] > 0 or bounds[1] < 0:
                return False
    except _Unsupported:
        return None

    # Try exact reduction and principal-root selection before the more
    # expensive invertibility certificate. Each has a separate work limit.
    for test in (lambda e: _relation_zero(e, only_zero=True),
            _isolated_radical_zero, _relation_zero):
        try:
            result = test(expr)
        except _Unsupported:
            continue
        if result is not None:
            return result
    return None
