"""Compact exponent-set arithmetic for univariate polynomial expressions.

``ExponentRuns`` represents a finite set of nonnegative exponents as a union
of pairwise-disjoint inclusive arithmetic progressions ``(start, stop, step)``.
The run algebra mirrors polynomial structure:

* union models the possible support of an ``Add``;
* Minkowski sum models the possible support of a ``Mul``;
* repeated Minkowski sum models the possible support of a nonnegative
  integer ``Pow``.

The expression walker :func:`exponent_runs` computes *structural* (possible)
exponent support.  It deliberately does not try to prove that symbolic
coefficients cancel; consumers such as a term stream can resolve those
coefficients separately.
"""

from collections import defaultdict
from math import gcd, isqrt
from operator import index

from sympy.core.sympify import sympify
from sympy.polys.polyerrors import PolynomialError


__all__ = ["ExponentRuns", "exponent_runs"]


def _validate_run(run):
    """Validate an inclusive arithmetic run."""
    start, stop, step = run
    if step <= 0:
        raise ValueError("step must be positive")
    if start > stop or (stop - start) % step:
        raise ValueError(f"invalid run: {run}")


def _run_len(run):
    start, stop, step = run
    return (stop - start)//step + 1


def _normalize_runs(runs):
    """Merge compatible runs and canonicalize singleton strides."""
    groups = defaultdict(list)

    for run in runs:
        _validate_run(run)
        start, stop, step = run

        if start == stop:
            step = 1

        groups[(step, start % step)].append((start, stop, step))

    result = []

    for group in groups.values():
        group.sort()
        start, stop, step = group[0]

        for a, b, _ in group[1:]:
            if a <= stop + step:
                stop = max(stop, b)
            else:
                result.append((start, stop, step))
                start, stop = a, b

        result.append((start, stop, step))

    return sorted(result)


def _proper_divisors(n):
    """Return the positive divisors of ``n`` smaller than ``n``."""
    if n <= 1:
        return []

    small = []
    large = []

    for d in range(1, isqrt(n) + 1):
        if n % d:
            continue

        q = n // d
        if d < n:
            small.append(d)
        if q != d and q < n:
            large.append(q)

    return small + large[::-1]


def _zip_group(group, step, newstep, residue):
    """Zip a complete residue family from ``step`` to ``newstep``."""
    width = step // newstep
    data = {}

    # In coordinates x = residue + newstep*j, the old stride is ``width``.
    for start, stop, _ in group:
        a = (start - residue) // newstep
        b = (stop - residue) // newstep
        r = a % width

        if r in data:
            # More than one disjoint run for one residue.  Leave it for a
            # later pass rather than trying to choose among several cores.
            return None

        data[r] = (a, b)

    if len(data) != width:
        return None

    lo = max(a for a, _ in data.values())
    hi = min(b for _, b in data.values())

    if lo > hi:
        return None

    def covered(j):
        a, b = data[j % width]
        return a <= j <= b

    while covered(lo - 1):
        lo -= 1
    while covered(hi + 1):
        hi += 1

    zip_lo = residue + newstep*lo
    zip_hi = residue + newstep*hi
    result = []

    for start, stop, _ in group:
        if start < zip_lo:
            left_stop = start + ((zip_lo - 1 - start)//step)*step
            if left_stop >= start:
                result.append((start, left_stop, step))

        if stop > zip_hi:
            right_start = start + (
                (zip_hi + 1 - start + step - 1)//step
            )*step
            if right_start <= stop:
                result.append((right_start, stop, step))

    result.append((zip_lo, zip_hi, newstep))
    return result


def _zip_runs(runs):
    """Recursively combine complete residue families into smaller strides."""
    runs = _normalize_runs(runs)

    while True:
        changed = False
        steps = sorted(
            {step for _, _, step in runs if step > 1},
            reverse=True,
        )

        for step in steps:
            same_step = [run for run in runs if run[2] == step]

            # One residue class cannot be zipped to a smaller stride.  Apart
            # from avoiding useless work, this is important when ``step`` is
            # large because even enumerating its proper divisors can be
            # expensive.
            if len(same_step) < 2:
                continue

            for newstep in _proper_divisors(step):
                width = step // newstep
                if len(same_step) < width:
                    continue

                buckets = defaultdict(list)
                for run in same_step:
                    buckets[run[0] % newstep].append(run)

                for residue, group in buckets.items():
                    # This bucket already contains only residues congruent to
                    # ``residue`` modulo ``newstep``.  There are exactly
                    # ``width`` such residue classes modulo ``step``, so a
                    # complete family is characterized by ``width`` distinct
                    # residues; there is no need to scan ``range(step)``.
                    residues = {start % step for start, _, _ in group}

                    if len(group) != width or len(residues) != width:
                        continue

                    replacement = _zip_group(
                        group, step, newstep, residue)
                    if replacement is None:
                        continue

                    selected = set(group)
                    runs = [run for run in runs if run not in selected]
                    runs.extend(replacement)
                    runs = _normalize_runs(runs)
                    changed = True
                    break

                if changed:
                    break
            if changed:
                break

        if not changed:
            return runs


def _run_intersection(A, B):
    """Return the arithmetic-run intersection of ``A`` and ``B``."""
    _validate_run(A)
    _validate_run(B)

    a0, a1, p = A
    b0, b1, q = B
    lo = max(a0, b0)
    hi = min(a1, b1)

    if lo > hi:
        return None

    g = gcd(p, q)
    if (b0 - a0) % g:
        return None

    p_reduced = p // g
    q_reduced = q // g
    step = p*q_reduced

    if q_reduced == 1:
        t = 0
    else:
        rhs = (b0 - a0) // g
        t = (rhs * pow(p_reduced, -1, q_reduced)) % q_reduced

    first = a0 + p*t
    if first < lo:
        first += ((lo - first + step - 1)//step)*step

    if first > hi:
        return None

    last = first + ((hi - first)//step)*step
    return (first, last, step)


def _run_difference(A, B):
    """Return disjoint runs representing the set difference ``A \\ B``."""
    intersection = _run_intersection(A, B)
    if intersection is None:
        return [A]

    a0, a1, p = A
    i0, i1, intersection_step = intersection
    last_index = (a1 - a0)//p
    first_removed = (i0 - a0)//p
    last_removed = (i1 - a0)//p

    # Every k-th point of A is removed between first_removed and last_removed.
    k = intersection_step // p
    result = []

    if first_removed:
        result.append((
            a0,
            a0 + p*(first_removed - 1),
            p,
        ))

    removed = (last_removed - first_removed)//k + 1

    if removed > 1 and k > 1:
        # Represent the interior either by the k - 1 surviving residue
        # classes or by the removed - 1 finite gaps, whichever is smaller.
        if k - 1 <= removed - 1:
            for r in range(1, k):
                start_index = first_removed + r
                if start_index > last_removed:
                    break

                stop_index = start_index + (
                    (last_removed - start_index)//k
                )*k
                result.append((
                    a0 + p*start_index,
                    a0 + p*stop_index,
                    intersection_step,
                ))
        else:
            removed_index = first_removed
            while removed_index + k <= last_removed:
                start_index = removed_index + 1
                stop_index = removed_index + k - 1
                result.append((
                    a0 + p*start_index,
                    a0 + p*stop_index,
                    p,
                ))
                removed_index += k

    if last_removed < last_index:
        result.append((
            a0 + p*(last_removed + 1),
            a1,
            p,
        ))

    return result


def _disjoint_runs(runs):
    """Return an exact pairwise-disjoint cover of ``runs``."""
    runs = _normalize_runs(runs)

    # Prefer broad, small-step runs so later runs are cut against them.
    runs.sort(key=lambda run: (run[2], run[0], -run[1]))
    result = []

    for run in runs:
        pieces = [run]

        for covered in result:
            new = []
            for piece in pieces:
                new.extend(_run_difference(piece, covered))
            pieces = new
            if not pieces:
                break

        result.extend(pieces)

    return _zip_runs(result)


def _run_sum(A, B):
    """Return the exact Minkowski sum of two arithmetic runs."""
    _validate_run(A)
    _validate_run(B)

    # The operation is commutative; canonicalizing the operands also makes
    # implementation choices independent of caller order.
    if B < A:
        A, B = B, A

    a0, a1, p = A
    b0, b1, q = B
    na = _run_len(A)
    nb = _run_len(B)

    if na == 1:
        return ExponentRuns([(a0 + b0, a0 + b1, q)])
    if nb == 1:
        return ExponentRuns([(a0 + b0, a1 + b0, p)])
    if p == q:
        return ExponentRuns([(a0 + b0, a1 + b1, p)])

    m = na - 1
    n = nb - 1
    g = gcd(p, q)
    p_reduced = p // g
    q_reduced = q // g
    candidates = []

    # Group shifts of A by B according to j mod p_reduced.  When A has at
    # least q_reduced points, shifts in one group touch or overlap on A's
    # p-lattice and collapse to one run.
    if na >= q_reduced:
        runs = []
        for r in range(min(p_reduced, nb)):
            last_j = r + p_reduced*((n - r)//p_reduced)
            runs.append((
                a0 + b0 + q*r,
                a1 + b0 + q*last_j,
                p,
            ))
        candidates.append(runs)

    # Symmetric construction, grouping shifts of B by A.
    if nb >= p_reduced:
        runs = []
        for r in range(min(q_reduced, na)):
            last_i = r + q_reduced*((m - r)//q_reduced)
            runs.append((
                a0 + b0 + p*r,
                a0 + b1 + p*last_i,
                q,
            ))
        candidates.append(runs)

    if candidates:
        return ExponentRuns(min(candidates, key=len))

    # If neither side is long enough for residue classes to merge, shift the
    # longer run by each point of the shorter run.  This creates at most
    # min(na, nb) runs rather than na*nb explicit values.
    if na <= nb:
        runs = [
            (a + b0, a + b1, q)
            for a in range(a0, a1 + 1, p)
        ]
    else:
        runs = [
            (a0 + b, a1 + b, p)
            for b in range(b0, b1 + 1, q)
        ]

    return ExponentRuns(runs)


def _format_run(run):
    start, stop, step = run

    if start == stop:
        return str(start)
    if step == 1:
        return f"[{start}, {stop}]"
    return f"[{start}, {stop}; {step}]"


class ExponentRuns:
    """Exact finite union of pairwise-disjoint arithmetic progressions.

    Each run is an inclusive ``(start, stop, step)`` tuple.  The constructor
    validates endpoints, removes overlap, and zips compatible residue classes.
    Iterating over an ``ExponentRuns`` object expands its represented values.
    """

    def __init__(self, runs=()):
        if isinstance(runs, ExponentRuns):
            self.runs = runs.runs
        else:
            self.runs = tuple(_disjoint_runs(runs))

    @classmethod
    def from_values(cls, values):
        return cls((value, value, 1) for value in values)

    def __repr__(self):
        return f"ExponentRuns({self.runs!r})"

    def __iter__(self):
        for start, stop, step in self.runs:
            yield from range(start, stop + 1, step)

    def __bool__(self):
        return bool(self.runs)

    def __len__(self):
        return self.size

    @property
    def size(self):
        """Number of represented exponents."""
        return sum(
            (stop - start)//step + 1
            for start, stop, step in self.runs
        )

    @property
    def lo(self):
        """Smallest represented exponent, or ``None`` for the empty set."""
        if not self.runs:
            return None
        return min(start for start, _, _ in self.runs)

    @property
    def hi(self):
        """Largest represented exponent, or ``None`` for the empty set."""
        if not self.runs:
            return None
        return max(stop for _, stop, _ in self.runs)

    @property
    def gap_count(self):
        if not self.runs:
            return 0
        return self.hi - self.lo + 1 - len(self)

    def __contains__(self, value):
        return any(
            start <= value <= stop and (value - start) % step == 0
            for start, stop, step in self.runs
        )

    def __or__(self, other):
        """Set union."""
        return ExponentRuns(self.runs + other.runs)

    def minkowski(self, other):
        """Minkowski sum of two exponent sets."""
        if not self.runs or not other.runs:
            return ExponentRuns()

        runs = []
        for A in self.runs:
            for B in other.runs:
                runs.extend(_run_sum(A, B).runs)

        return ExponentRuns(runs)

    __add__ = minkowski

    def _pow(self, n):
        result = ExponentRuns([(0, 0, 1)])
        base = self

        while n:
            if n & 1:
                result = result + base
            n >>= 1
            if n:
                base = base + base

        return result

    def _affine(self, scale=1, shift=0):
        """Return ``scale*self + shift`` without re-normalizing runs.

        A positive integral scale and a translation preserve disjointness
        and the canonical run structure, so an already-normalized
        ``ExponentRuns`` can be transformed directly.
        """
        if scale <= 0:
            raise ValueError("positive scale expected")

        if scale == 1 and shift == 0:
            return self

        result = object.__new__(ExponentRuns)
        result.runs = tuple(
            (
                shift + scale*start,
                shift + scale*stop,
                scale*step if start != stop else 1,
            )
            for start, stop, step in self.runs
        )
        return result

    def _shift(self, amount):
        """Return this exponent set translated by ``amount``."""
        return self._affine(shift=amount)

    def _stable_power(self, n):
        """Return the n-fold Minkowski sum, extrapolating stable powers.

        This is called only for a normalized support: its minimum is 0 and
        its lattice gcd is 1.

        The ordinary binary-power walk naturally constructs

            A, 2*A, 4*A, 8*A, ...

        so stabilization is checked only at those already-needed powers.
        If ``S = k*A`` and ``b = max(A)``, stability gives

            (k + 1)*A = S | (S + b),

        after which every later power is obtained from ``S`` by adding
        multiples of ``b``.

        Before checking that recurrence, use the cheap necessary signal that
        doubling ``S`` increased its cardinality by exactly ``k*b``.  The
        b-l theorem is used only when the binary walk naturally reaches its
        bound; no large intermediate is constructed merely to reach it.
        """
        if n <= 1 or len(self) <= 1:
            return self._pow(n)

        b = self.hi
        if not b:
            return self._pow(n)

        # For A = {0 = a0 < ... < a_(l+1) = b}, l = len(A) - 2.
        bound = max(1, b - len(self) + 2)

        def extrapolate(S, k):
            if k == n:
                return S

            shift = (n - k)*b
            runs = list(S.runs)

            # In the stable regime there is a central contiguous run at
            # least b terms long. Later powers leave the low fringe fixed,
            # extend this middle run, and translate the high fringe.
            for i, (start, stop, step) in enumerate(runs):
                if step == 1 and stop - start + 1 >= b:
                    return ExponentRuns(
                        runs[:i]
                        + [(start, stop + shift, 1)]
                        + [
                            (a + shift, z + shift, s)
                            for a, z, s in runs[i + 1:]
                        ]
                    )

            # Defensive exact fallback if a representation reaches here
            # without the expected stable middle run.
            return S + ExponentRuns([(0, shift, b)])

        result = ExponentRuns([(0, 0, 1)])
        base = self
        k = 1
        remaining = n

        while remaining:
            # If the normal binary walk has naturally reached the theorem
            # bound, ``base = k*A`` is already in the stable regime.
            if k >= bound:
                return extrapolate(base, k)

            if remaining & 1:
                result = result + base

            remaining >>= 1
            if not remaining:
                break

            next_base = base + base

            # Equal gap counts between k*A and 2k*A are a cheap signal that
            # stabilization may already have occurred:
            #
            #     |2k*A| - |k*A| == k*b.
            #
            # Only then pay to verify the stronger adjacent recurrence.
            if len(next_base) - len(base) == k*b:
                adjacent = base + self
                fringe = base | base._shift(b)

                # ``fringe`` is a subset of ``adjacent`` because 0 and b
                # belong to A. Equal sizes therefore prove equality.
                if len(adjacent) == len(fringe):
                    return extrapolate(base, k)

            base = next_base
            k *= 2

        return result

    def __pow__(self, n):
        if n < 0:
            raise ValueError("nonnegative power expected")

        if n == 0:
            return ExponentRuns([(0, 0, 1)])

        if not self.runs:
            return ExponentRuns()

        lo = self.lo
        g = self.lattice_gcd

        if lo == 0 and g == 1:
            return self._stable_power(n)

        normalized = ExponentRuns(
            (
                (start - lo)//g,
                (stop - lo)//g,
                step//g if start != stop else 1,
            )
            for start, stop, step in self.runs
        )

        result = normalized._stable_power(n)
        return result._affine(g, n*lo)

    def is_direct_sum(self, other):
        """Return whether the Minkowski sum with ``other`` is direct."""
        if not self or not other:
            return True
        return len(self + other) == len(self)*len(other)

    @property
    def lattice_gcd(self):
        """Return the common lattice spacing of this support."""
        if not self.runs:
            return 1

        lo = self.lo
        g = 0

        for start, stop, step in self.runs:
            g = gcd(g, start - lo)
            if start != stop:
                g = gcd(g, step)

        return g or 1

    def summary(self, max_runs=12):
        """Return a compact description of this exponent support."""
        if not self.runs:
            return "ExponentRuns(empty)"

        count = len(self)
        span = self.hi - self.lo + 1
        gaps = span - count

        shown = list(self.runs[:max_runs])
        pieces = [_format_run(run) for run in shown]

        if len(self.runs) > max_runs:
            pieces.append(f"... ({len(self.runs) - max_runs} more)")

        support = ", ".join(pieces)

        return (
            f"ExponentRuns(\n"
            f"  lo={self.lo}, hi={self.hi},\n"
            f"  count={count}, span={span}, gaps={gaps},\n"
            f"  runs={len(self.runs)},\n"
            f"  support={support}\n"
            f")"
        )

def exponent_runs(expr, x):
    """Return compact structural exponent support for ``expr`` in ``x``.

    The result contains every degree that can occur from the expression
    structure.  Addition takes the union of operand supports, so coefficient
    cancellation can make the actual polynomial support smaller.
    """
    expr = sympify(expr)
    gen = sympify(x)

    b, n = expr.as_base_exp()
    if b == gen and n.is_Integer and n >= 0:
        n = int(n)
        return ExponentRuns([(n, n, 1)])

    if expr.is_zero is True:
        return ExponentRuns()

    if not expr.has(gen):
        return ExponentRuns([(0, 0, 1)])

    if expr.is_Add:
        result = ExponentRuns()
        for arg in expr.args:
            result |= exponent_runs(arg, gen)
        return result

    if expr.is_Mul:
        result = ExponentRuns([(0, 0, 1)])
        for arg in expr.args:
            result += exponent_runs(arg, gen)
        return result

    if expr.is_Pow and expr.exp.is_Integer and expr.exp >= 0:
        return exponent_runs(expr.base, gen) ** int(expr.exp)

    raise PolynomialError(f"{expr} is not a polynomial in {gen}")
