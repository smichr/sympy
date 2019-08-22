from __future__ import print_function, division

from sympy.core.basic import Basic
from sympy.core.compatibility import as_int, with_metaclass, range, PY3
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import oo
from sympy.core.relational import Eq, Ge, Le, Ne, Gt, Lt
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, converter
from sympy.logic.boolalg import And, Or
from sympy.sets.sets import (Set, Interval, Union, FiniteSet,
    ProductSet, Intersection)
from sympy.utilities.misc import filldedent


class Rationals(with_metaclass(Singleton, Set)):
    """
    Represents the rational numbers. This set is also available as
    the Singleton, S.Rationals.

    Examples
    ========

    >>> from sympy import S
    >>> S.Half in S.Rationals
    True
    >>> iterable = iter(S.Rationals)
    >>> [next(iterable) for i in range(12)]
    [0, 1, -1, 1/2, 2, -1/2, -2, 1/3, 3, -1/3, -3, 2/3]
    """

    is_iterable = True
    _inf = S.NegativeInfinity
    _sup = S.Infinity

    def _contains(self, other):
        if not isinstance(other, Expr):
            return False
        if other.is_Number:
            return other.is_Rational
        return other.is_rational

    def __iter__(self):
        from sympy.core.numbers import igcd, Rational
        yield S.Zero
        yield S.One
        yield S.NegativeOne
        d = 2
        while True:
            for n in range(d):
                if igcd(n, d) == 1:
                    yield Rational(n, d)
                    yield Rational(d, n)
                    yield Rational(-n, d)
                    yield Rational(-d, n)
            d += 1

    @property
    def _boundary(self):
        return self


class Naturals(with_metaclass(Singleton, Set)):
    """
    Represents the natural numbers (or counting numbers) which are all
    positive integers starting from 1. This set is also available as
    the Singleton, S.Naturals.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Naturals)
    >>> next(iterable)
    1
    >>> next(iterable)
    2
    >>> next(iterable)
    3
    >>> pprint(S.Naturals.intersect(Interval(0, 10)))
    {1, 2, ..., 10}

    See Also
    ========

    Naturals0 : non-negative integers (i.e. includes 0, too)
    Integers : also includes negative integers
    """

    is_iterable = True
    _inf = S.One
    _sup = S.Infinity

    def _contains(self, other):
        if not isinstance(other, Expr):
            return False
        elif other.is_positive and other.is_integer:
            return True
        elif other.is_integer is False or other.is_positive is False:
            return False

    def __iter__(self):
        i = self._inf
        while True:
            yield i
            i = i + 1

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        from sympy.functions.elementary.integers import floor
        return And(Eq(floor(x), x), x >= self.inf, x < oo)


class Naturals0(Naturals):
    """Represents the whole numbers which are all the non-negative integers,
    inclusive of zero.

    See Also
    ========

    Naturals : positive integers; does not include 0
    Integers : also includes the negative integers
    """
    _inf = S.Zero

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_integer and other.is_nonnegative:
            return S.true
        elif other.is_integer is False or other.is_nonnegative is False:
            return S.false


class Integers(with_metaclass(Singleton, Set)):
    """
    Represents all integers: positive, negative and zero. This set is also
    available as the Singleton, S.Integers.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Integers)
    >>> next(iterable)
    0
    >>> next(iterable)
    1
    >>> next(iterable)
    -1
    >>> next(iterable)
    2

    >>> pprint(S.Integers.intersect(Interval(-4, 4)))
    {-4, -3, ..., 4}

    See Also
    ========

    Naturals0 : non-negative integers
    Integers : positive and negative integers and zero
    """

    is_iterable = True

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        return other.is_integer

    def __iter__(self):
        yield S.Zero
        i = S.One
        while True:
            yield i
            yield -i
            i = i + 1

    @property
    def _inf(self):
        return -S.Infinity

    @property
    def _sup(self):
        return S.Infinity

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        from sympy.functions.elementary.integers import floor
        return And(Eq(floor(x), x), -oo < x, x < oo)


class Reals(with_metaclass(Singleton, Interval)):
    """
    Represents all real numbers
    from negative infinity to positive infinity,
    including all integer, rational and irrational numbers.
    This set is also available as the Singleton, S.Reals.


    Examples
    ========

    >>> from sympy import S, Interval, Rational, pi, I
    >>> 5 in S.Reals
    True
    >>> Rational(-1, 2) in S.Reals
    True
    >>> pi in S.Reals
    True
    >>> 3*I in S.Reals
    False
    >>> S.Reals.contains(pi)
    True


    See Also
    ========

    ComplexRegion
    """
    def __new__(cls):
        return Interval.__new__(cls, -S.Infinity, S.Infinity)

    def __eq__(self, other):
        return other == Interval(-S.Infinity, S.Infinity)

    def __hash__(self):
        return hash(Interval(-S.Infinity, S.Infinity))


class ImageSet(Set):
    """
    Image of a set under a mathematical function. The transformation
    must be given as a Lambda function which has as many arguments
    as the elements of the set upon which it operates, e.g. 1 argument
    when acting on the set of integers or 2 arguments when acting on
    a complex region.

    This function is not normally called directly, but is called
    from `imageset`.


    Examples
    ========

    >>> from sympy import Symbol, S, pi, Dummy, Lambda
    >>> from sympy.sets.sets import FiniteSet, Interval
    >>> from sympy.sets.fancysets import ImageSet

    >>> x = Symbol('x')
    >>> N = S.Naturals
    >>> squares = ImageSet(Lambda(x, x**2), N) # {x**2 for x in N}
    >>> 4 in squares
    True
    >>> 5 in squares
    False

    >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
    {1, 4, 9}

    >>> square_iterable = iter(squares)
    >>> for i in range(4):
    ...     next(square_iterable)
    1
    4
    9
    16

    If you want to get value for `x` = 2, 1/2 etc. (Please check whether the
    `x` value is in `base_set` or not before passing it as args)

    >>> squares.lamda(2)
    4
    >>> squares.lamda(S(1)/2)
    1/4

    >>> n = Dummy('n')
    >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
    >>> dom = Interval(-1, 1)
    >>> dom.intersect(solutions)
    {0}

    See Also
    ========

    sympy.sets.sets.imageset
    """
    def __new__(cls, flambda, *sets):
        if not isinstance(flambda, Lambda):
            raise ValueError('first argument must be a Lambda')

        if flambda is S.IdentityFunction:
            if len(sets) != 1:
                raise ValueError('identify function requires a single set')
            return sets[0]

        if not set(flambda.variables) & flambda.expr.free_symbols:
            return FiniteSet(flambda.expr)

        return Basic.__new__(cls, flambda, *sets)

    lamda = property(lambda self: self.args[0])
    base_set = property(lambda self: ProductSet(self.args[1:]))

    def __iter__(self):
        already_seen = set()
        for i in self.base_set:
            val = self.lamda(i)
            if val in already_seen:
                continue
            else:
                already_seen.add(val)
                yield val

    def _is_multivariate(self):
        return len(self.lamda.variables) > 1

    def _contains(self, other):
        from sympy.matrices import Matrix
        from sympy.solvers.solveset import solveset, linsolve
        from sympy.solvers.solvers import solve
        from sympy.utilities.iterables import is_sequence, iterable, cartes
        L = self.lamda
        if is_sequence(other) != is_sequence(L.expr):
            return False
        elif is_sequence(other) and len(L.expr) != len(other):
            return False

        if self._is_multivariate():
            if not is_sequence(L.expr):
                # exprs -> (numer, denom) and check again
                # XXX this is a bad idea -- make the user
                # remap self to desired form
                return other.as_numer_denom() in self.func(
                    Lambda(L.variables, L.expr.as_numer_denom()), self.base_set)
            eqs = [expr - val for val, expr in zip(other, L.expr)]
            variables = L.variables
            free = set(variables)
            if all(i.is_number for i in list(Matrix(eqs).jacobian(variables))):
                solns = list(linsolve([e - val for e, val in
                zip(L.expr, other)], variables))
            else:
                try:
                    syms = [e.free_symbols & free for e in eqs]
                    solns = {}
                    for i, (e, s, v) in enumerate(zip(eqs, syms, other)):
                        if not s:
                            if e != v:
                                return S.false
                            solns[vars[i]] = [v]
                            continue
                        elif len(s) == 1:
                            sy = s.pop()
                            sol = solveset(e, sy)
                            if sol is S.EmptySet:
                                return S.false
                            elif isinstance(sol, FiniteSet):
                                solns[sy] = list(sol)
                            else:
                                raise NotImplementedError
                        else:
                            # if there is more than 1 symbol from
                            # variables in expr than this is a
                            # coupled system
                            raise NotImplementedError
                    solns = cartes(*[solns[s] for s in variables])
                except NotImplementedError:
                    solns = solve([e - val for e, val in
                        zip(L.expr, other)], variables, set=True)
                    if solns:
                        _v, solns = solns
                        # watch for infinite solutions like solving
                        # for x, y and getting (x, 0), (0, y), (0, 0)
                        solns = [i for i in solns if not any(
                            s in i for s in variables)]
                        if not solns:
                            return False
                    else:
                        # not sure if [] means no solution or
                        # couldn't find one
                        return
        else:
            x = L.variables[0]
            if isinstance(L.expr, Expr):
                # scalar -> scalar mapping
                solnsSet = solveset(L.expr - other, x)
                if solnsSet.is_FiniteSet:
                    solns = list(solnsSet)
                else:
                    msgset = solnsSet
            else:
                # scalar -> vector
                # note: it is not necessary for components of other
                # to be in the corresponding base set unless the
                # computed component is always in the corresponding
                # domain. e.g. 1/2 is in imageset(x, x/2, Integers)
                # while it cannot be in imageset(x, x + 2, Integers).
                # So when the base set is comprised of integers or reals
                # perhaps a pre-check could be done to see if the computed
                # values are still in the set.
                dom = self.base_set
                for e, o in zip(L.expr, other):
                    msgset = dom
                    other = e - o
                    dom = dom.intersection(solveset(e - o, x, domain=dom))
                    if not dom:
                        # there is no solution in common
                        return False
                return not isinstance(dom, Intersection)
        for soln in solns:
            try:
                if soln in self.base_set:
                    return True
            except TypeError:
                return
        return S.false

    @property
    def is_iterable(self):
        return self.base_set.is_iterable

    def doit(self, **kwargs):
        from sympy.sets.setexpr import SetExpr
        f = self.lamda
        base_set = self.base_set
        return SetExpr(base_set)._eval_func(f).set


class Range(Set):
    """
    Represents a range of integers. Can be called as Range(stop),
    Range(start, stop), or Range(start, stop, step); when stop is
    not given it defaults to 1.

    `Range(stop)` is the same as `Range(0, stop, 1)` and the stop value
    (juse as for Python ranges) is not included in the Range values.

        >>> from sympy import Range, oo
        >>> list(Range(3))
        [0, 1, 2]

    The step can also be negative:

        >>> list(Range(10, 0, -2))
        [10, 8, 6, 4, 2]

    The stop value is made canonical so equivalent ranges always
    have the same args:

        >>> Range(1, -oo, 2)
        Range(0, 0, 1)
        >>> Range(1, -oo, 3)
        Range(0, 0, 1)

    Infinite ranges are allowed. ``oo`` and ``-oo`` are never included in the
    set (``Range`` is always a subset of ``Integers``). If the starting point
    is infinite, then the final value is ``stop - step``. To iterate such a
    range, it needs to be reversed:

        >>> from sympy import oo
        >>> r = Range(-oo, 1)
        >>> r[-1]
        0
        >>> next(iter(r))
        Traceback (most recent call last):
        ...
        ValueError: Cannot iterate over Range with infinite start
        >>> next(iter(r.reversed))
        0

    Although Range is a set (and supports the normal set
    operations) it maintains the order of the elements and can
    be used in contexts where `range` would be used.

        >>> from sympy import Interval
        >>> Range(0, 10, 2).intersect(Interval(3, 7))
        Range(4, 8, 2)
        >>> list(_)
        [4, 6]

    Although slicing of a Range will always return a Range -- possibly
    empty -- an empty set will be returned from any intersection that
    is empty:

        >>> Range(3)[:0]
        Range(0, 0, 1)
        >>> Range(3).intersect(Interval(4, oo))
        EmptySet()
        >>> Range(3).intersect(Range(4, oo))
        EmptySet()

    Range also supports symbolic start, stop and step provided they
    satisfy, is_integer=True, i.e., all the paramters should be integer
    symbols:

        >>> from sympy import symbols
        >>> n = symbols('n', integer=True)
        >>> Range(n, 10, 1)
        Range(n, 10, 1)
        >>> list(Range(n, n + 6))
        [n, n + 1, n + 2, n + 3, n + 4, n + 5]
        >>> Range(1, n, 1).size
        Piecewise((Abs(n - 1), n - 1 > 0), (0, True))
    """
    is_iterable = True

    def __new__(cls, *args):
        from sympy.functions.elementary.integers import ceiling
        if len(args) == 1:
            if isinstance(args[0], range):
                args = args[0].__reduce__()[1]  # use pickle method

        # expand range
        slc = slice(*args)

        if slc.step == 0:
            raise ValueError("step cannot be 0")

        start, stop, step = slc.start or 0, slc.stop, slc.step or 1
        params = []
        for w in (start, stop, step):
            if (w in [S.NegativeInfinity, S.Infinity]) or (sympify(w).is_integer == True):
                params.append(sympify(w))
            else:
                raise ValueError(filldedent('''
        Arguments to Range must be integers (or integer symbols); `imageset` can define
        other cases, e.g. use `imageset(i, i/10, Range(3))` to give
        [0, 1/10, 1/5].'''))

        start, stop, step = params

        if not step.is_finite == True:
            raise ValueError("step must be a finite integer symbol.")

        if all(i.is_infinite for i in  (start, stop)):
            if start == stop:
                # canonical null handled below
                start = stop = S.One
            else:
                raise ValueError(filldedent('''
    Either the start or end value of the Range must be finite.'''))

        if not step.is_Symbol:
            if start.is_infinite:
                if step*(stop - start) < 0:
                    start = stop = S.One
                else:
                    end = stop
            if not start.is_infinite:
                ref = start if start.is_finite else stop
                n = ceiling((stop - ref)/step)
                if (n <= 0) == True:
                    # null Range
                    start = end = S.Zero
                    step = S.One
                else:
                    end = ref + n*step
        else:
            end = stop

        obj = Basic.__new__(cls, start, end, step)
        return obj

    start = property(lambda self: self.args[0])
    stop = property(lambda self: self.args[1])
    step = property(lambda self: self.args[2])

    @property
    def reversed(self):
        from sympy.functions.elementary.integers import ceiling
        from sympy.functions.elementary.piecewise import Piecewise
        """Return an equivalent Range in the opposite order.

        Examples
        ========

        >>> from sympy import Range
        >>> Range(10).reversed
        Range(9, -1, -1)
        """
        if not self:
            return self
        start, stop, step = self.start, self.stop, self.step
        return self.func(stop - step, start - step, -step)

    def _contains(self, other):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.elementary.integers import ceiling
        other = _sympify(other)
        if not self:
            return S.false
        if other.is_infinite:
            return S.false
        if not other.is_integer:
            return other.is_integer
        ref = self.start if self.start.is_finite else self.stop
        start, stop, step = self.start, self.stop, self.step
        inf = Piecewise((start, Ge(step, S.Zero)), (stop - step, True))
        sup = Piecewise((stop - step, Ge(step, S.Zero)), (start, True))
        return Piecewise((False, Le(ceiling((stop - start)/step), S.Zero)),
                         (False, Ne((ref - other) % self.step, S.Zero)),
                         (True, Ge(other, inf) & Le(other, sup)),
                         (False, True))

    def __iter__(self):
        if self.start in [S.NegativeInfinity, S.Infinity]:
            raise ValueError("Cannot iterate over Range with infinite start")
        if not (self.size.is_Integer or self.size in [S.NegativeInfinity, S.Infinity]):
            raise ValueError("Cannot iterate over symbolic sized range.")
        if self:
            i = self.start
            step = self.step

            while And(Or(Le(step, 0), And(Le(self.start, i), Lt(i, self.stop))),
                     Or(Ge(step, 0), And(Lt(self.stop, i), Le(i, self.start)))) != False:
                yield i
                i += step

    def __len__(self):
        size = _sympify(self.size)
        if size.has(Symbol) or size.is_infinite:
            raise ValueError(
                "Use .size to get the length of an infinite and symbolic Range")
        return int(size)

    @property
    def size(self):
        from sympy.functions.elementary.integers import floor, ceiling
        from sympy.functions.elementary.piecewise import Piecewise
        if not self:
            return S.Zero
        start, stop, step = self.start, self.stop, self.step
        dif = stop - start
        if dif.is_infinite:
            return S.Infinity
        null = ceiling(dif/step)
        return Piecewise((abs(floor(dif/step)), Gt(null, S.Zero)),
                         (S.Zero, True))

    def __nonzero__(self):
        return self.start != self.stop

    __bool__ = __nonzero__

    def __getitem__(self, i):
        from sympy.functions.elementary.integers import ceiling
        from sympy.functions.elementary.piecewise import Piecewise
        ooslice = "cannot slice from the end with an infinite value"
        if isinstance(i, slice):
            if self.size.has(Symbol):
                raise NotImplementedError("Cannot slice Range of symbolic sizes.")
            if self.size == S.Zero:
                return Range(0)
            istart, istop, istep = i.start, i.stop, i.step
            if any(isinstance(par, Symbol) for par in (istart, istop, istep)):
                raise NotImplementedError("Range cannot be sliced because "
                                          "slice has symbolic start, stop, or step.")
            bound_check_f = lambda x: x is not None and \
                                      ((Ge(x, self.size) == True) or
                                      (Lt(x, -self.size) == True))
            if istep != None and istep < 0:
                if bound_check_f(istart):
                    istart = self.size - 1
                if bound_check_f(istop):
                    istop = self.size - 1
                if istart is None:
                    istart = 0
                else:
                    istart = -(istart + 1)
                istop = -(istop + 1) if istop is not None else istop
                return self.reversed[istart:istop:-istep]
            if bound_check_f(istart):
                return Range(0)
            start = self[0] if (istart == None) else self[istart]
            step = self.step if (istep == None) else istep * self.step
            if istop is None:
                stop = self.stop
            else:
                bound_check = Or(And(Lt(istop, self.size), Ge(istop, S.Zero)),
                                And(Ge(istop, -self.size), Le(istop, -1)))
                if bound_check == False:
                    stop = self.stop
                else:
                    stop = Piecewise((self[istop], bound_check), (self.stop, True))
            return Range(start, stop, step)
        else:
            i = _sympify(i)
            if not self or ((i.is_integer == False) and (i.is_infinite == False)):
                raise IndexError('Range index out of range')
            if i == S(0):
                return Piecewise((self._inf, Gt(self.step, 0)), (self._sup, True))
            if i == -S(1) or i is S.Infinity:
                return Piecewise((self._sup, Gt(self.step, 0)), (self._inf, True))
            start, stop, step = self.start, self.stop, self.step
            rvstop, rvstart = (stop + i*step, start + i*step)
            rv = Piecewise((rvstop, Lt(i, 0)), (rvstart, True))
            bound = Or(And(Or(Lt(rvstop, self._inf), Gt(rvstop, self._sup)), Lt(i, 0)),
                  And(Or(Lt(rvstart, self._inf), Gt(rvstart, self._sup)), Ge(i, 0)))
            if bound == True:
                raise IndexError("Range index out of range")
            return rv

    @property
    def _inf(self):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.elementary.integers import ceiling
        if not self:
            raise NotImplementedError
        start, stop, step = self.start, self.stop, self.step
        return Piecewise((S.NaN, Le(ceiling((stop - start)/step), S.Zero)),
                        (start, Ge(step, S.Zero)),
                        (stop - step, True))

    @property
    def _sup(self):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.elementary.integers import ceiling
        if not self:
            raise NotImplementedError
        start, stop, step = self.start, self.stop, self.step
        return Piecewise((S.NaN, Le(ceiling((stop - start)/step), S.Zero)),
                        (stop - step, Ge(step, S.Zero)),
                        (start, True))

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        """Rewrite a Range in terms of equalities and logic operators. """
        from sympy.functions.elementary.integers import floor
        i = (x - (self.inf if self.inf.is_finite else self.sup))/self.step
        return And(
            Eq(i, floor(i)),
            x >= self.inf if self.inf in self else x > self.inf,
            x <= self.sup if self.sup in self else x < self.sup)

converter[range] = Range

def normalize_theta_set(theta):
    """
    Normalize a Real Set `theta` in the Interval [0, 2*pi). It returns
    a normalized value of theta in the Set. For Interval, a maximum of
    one cycle [0, 2*pi], is returned i.e. for theta equal to [0, 10*pi],
    returned normalized value would be [0, 2*pi). As of now intervals
    with end points as non-multiples of `pi` is not supported.

    Raises
    ======

    NotImplementedError
        The algorithms for Normalizing theta Set are not yet
        implemented.
    ValueError
        The input is not valid, i.e. the input is not a real set.
    RuntimeError
        It is a bug, please report to the github issue tracker.

    Examples
    ========

    >>> from sympy.sets.fancysets import normalize_theta_set
    >>> from sympy import Interval, FiniteSet, pi
    >>> normalize_theta_set(Interval(9*pi/2, 5*pi))
    Interval(pi/2, pi)
    >>> normalize_theta_set(Interval(-3*pi/2, pi/2))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-pi/2, pi/2))
    Union(Interval(0, pi/2), Interval.Ropen(3*pi/2, 2*pi))
    >>> normalize_theta_set(Interval(-4*pi, 3*pi))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-3*pi/2, -pi/2))
    Interval(pi/2, 3*pi/2)
    >>> normalize_theta_set(FiniteSet(0, pi, 3*pi))
    {0, pi}

    """
    from sympy.functions.elementary.trigonometric import _pi_coeff as coeff

    if theta.is_Interval:
        interval_len = theta.measure
        # one complete circle
        if interval_len >= 2*S.Pi:
            if interval_len == 2*S.Pi and theta.left_open and theta.right_open:
                k = coeff(theta.start)
                return Union(Interval(0, k*S.Pi, False, True),
                        Interval(k*S.Pi, 2*S.Pi, True, True))
            return Interval(0, 2*S.Pi, False, True)

        k_start, k_end = coeff(theta.start), coeff(theta.end)

        if k_start is None or k_end is None:
            raise NotImplementedError("Normalizing theta without pi as coefficient is "
                                    "not yet implemented")
        new_start = k_start*S.Pi
        new_end = k_end*S.Pi

        if new_start > new_end:
            return Union(Interval(S.Zero, new_end, False, theta.right_open),
                         Interval(new_start, 2*S.Pi, theta.left_open, True))
        else:
            return Interval(new_start, new_end, theta.left_open, theta.right_open)

    elif theta.is_FiniteSet:
        new_theta = []
        for element in theta:
            k = coeff(element)
            if k is None:
                raise NotImplementedError('Normalizing theta without pi as '
                                          'coefficient, is not Implemented.')
            else:
                new_theta.append(k*S.Pi)
        return FiniteSet(*new_theta)

    elif theta.is_Union:
        return Union(*[normalize_theta_set(interval) for interval in theta.args])

    elif theta.is_subset(S.Reals):
        raise NotImplementedError("Normalizing theta when, it is of type %s is not "
                                  "implemented" % type(theta))
    else:
        raise ValueError(" %s is not a real set" % (theta))


class ComplexRegion(Set):
    """
    Represents the Set of all Complex Numbers. It can represent a
    region of Complex Plane in both the standard forms Polar and
    Rectangular coordinates.

    * Polar Form
      Input is in the form of the ProductSet or Union of ProductSets
      of the intervals of r and theta, & use the flag polar=True.

    Z = {z in C | z = r*[cos(theta) + I*sin(theta)], r in [r], theta in [theta]}

    * Rectangular Form
      Input is in the form of the ProductSet or Union of ProductSets
      of interval of x and y the of the Complex numbers in a Plane.
      Default input type is in rectangular form.

    Z = {z in C | z = x + I*y, x in [Re(z)], y in [Im(z)]}

    Examples
    ========

    >>> from sympy.sets.fancysets import ComplexRegion
    >>> from sympy.sets import Interval
    >>> from sympy import S, I, Union
    >>> a = Interval(2, 3)
    >>> b = Interval(4, 6)
    >>> c = Interval(1, 8)
    >>> c1 = ComplexRegion(a*b)  # Rectangular Form
    >>> c1
    ComplexRegion(ProductSet(Interval(2, 3), Interval(4, 6)), False)

    * c1 represents the rectangular region in complex plane
      surrounded by the coordinates (2, 4), (3, 4), (3, 6) and
      (2, 6), of the four vertices.

    >>> c2 = ComplexRegion(Union(a*b, b*c))
    >>> c2
    ComplexRegion(Union(ProductSet(Interval(2, 3), Interval(4, 6)), ProductSet(Interval(4, 6), Interval(1, 8))), False)

    * c2 represents the Union of two rectangular regions in complex
      plane. One of them surrounded by the coordinates of c1 and
      other surrounded by the coordinates (4, 1), (6, 1), (6, 8) and
      (4, 8).

    >>> 2.5 + 4.5*I in c1
    True
    >>> 2.5 + 6.5*I in c1
    False

    >>> r = Interval(0, 1)
    >>> theta = Interval(0, 2*S.Pi)
    >>> c2 = ComplexRegion(r*theta, polar=True)  # Polar Form
    >>> c2  # unit Disk
    ComplexRegion(ProductSet(Interval(0, 1), Interval.Ropen(0, 2*pi)), True)

    * c2 represents the region in complex plane inside the
      Unit Disk centered at the origin.

    >>> 0.5 + 0.5*I in c2
    True
    >>> 1 + 2*I in c2
    False

    >>> unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    >>> upper_half_unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    >>> intersection = unit_disk.intersect(upper_half_unit_disk)
    >>> intersection
    ComplexRegion(ProductSet(Interval(0, 1), Interval(0, pi)), True)
    >>> intersection == upper_half_unit_disk
    True

    See Also
    ========

    Reals

    """
    is_ComplexRegion = True

    def __new__(cls, sets, polar=False):
        from sympy import sin, cos

        x, y, r, theta = symbols('x, y, r, theta', cls=Dummy)
        I = S.ImaginaryUnit
        polar = sympify(polar)

        # Rectangular Form
        if polar == False:
            if all(_a.is_FiniteSet for _a in sets.args) and (len(sets.args) == 2):

                # ** ProductSet of FiniteSets in the Complex Plane. **
                # For Cases like ComplexRegion({2, 4}*{3}), It
                # would return {2 + 3*I, 4 + 3*I}
                complex_num = []
                for x in sets.args[0]:
                    for y in sets.args[1]:
                        complex_num.append(x + I*y)
                obj = FiniteSet(*complex_num)
            else:
                obj = ImageSet.__new__(cls, Lambda((x, y), x + I*y), sets)
            obj._variables = (x, y)
            obj._expr = x + I*y

        # Polar Form
        elif polar == True:
            new_sets = []
            # sets is Union of ProductSets
            if not sets.is_ProductSet:
                for k in sets.args:
                    new_sets.append(k)
            # sets is ProductSets
            else:
                new_sets.append(sets)
            # Normalize input theta
            for k, v in enumerate(new_sets):
                new_sets[k] = ProductSet(v.args[0],
                                         normalize_theta_set(v.args[1]))
            sets = Union(*new_sets)
            obj = ImageSet.__new__(cls, Lambda((r, theta),
                                   r*(cos(theta) + I*sin(theta))),
                                   sets)
            obj._variables = (r, theta)
            obj._expr = r*(cos(theta) + I*sin(theta))

        else:
            raise ValueError("polar should be either True or False")

        obj._sets = sets
        obj._polar = polar
        return obj

    @property
    def sets(self):
        """
        Return raw input sets to the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.sets
        ProductSet(Interval(2, 3), Interval(4, 5))
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.sets
        Union(ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        return self._sets

    @property
    def args(self):
        return (self._sets, self._polar)

    @property
    def variables(self):
        return self._variables

    @property
    def expr(self):
        return self._expr

    @property
    def psets(self):
        """
        Return a tuple of sets (ProductSets) input of the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)),)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        if self.sets.is_ProductSet:
            psets = ()
            psets = psets + (self.sets, )
        else:
            psets = self.sets.args
        return psets

    @property
    def a_interval(self):
        """
        Return the union of intervals of `x` when, self is in
        rectangular form, or the union of intervals of `r` when
        self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.a_interval
        Interval(2, 3)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.a_interval
        Union(Interval(2, 3), Interval(4, 5))

        """
        a_interval = []
        for element in self.psets:
            a_interval.append(element.args[0])

        a_interval = Union(*a_interval)
        return a_interval

    @property
    def b_interval(self):
        """
        Return the union of intervals of `y` when, self is in
        rectangular form, or the union of intervals of `theta`
        when self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.b_interval
        Interval(4, 5)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.b_interval
        Interval(1, 7)

        """
        b_interval = []
        for element in self.psets:
            b_interval.append(element.args[1])

        b_interval = Union(*b_interval)
        return b_interval

    @property
    def polar(self):
        """
        Returns True if self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union, S
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> theta = Interval(0, 2*S.Pi)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.polar
        False
        >>> C2 = ComplexRegion(a*theta, polar=True)
        >>> C2.polar
        True
        """
        return self._polar

    @property
    def _measure(self):
        """
        The measure of self.sets.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, S
        >>> a, b = Interval(2, 5), Interval(4, 8)
        >>> c = Interval(0, 2*S.Pi)
        >>> c1 = ComplexRegion(a*b)
        >>> c1.measure
        12
        >>> c2 = ComplexRegion(a*c, polar=True)
        >>> c2.measure
        6*pi

        """
        return self.sets._measure

    @classmethod
    def from_real(cls, sets):
        """
        Converts given subset of real numbers to a complex region.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion
        >>> unit = Interval(0,1)
        >>> ComplexRegion.from_real(unit)
        ComplexRegion(ProductSet(Interval(0, 1), {0}), False)

        """
        if not sets.is_subset(S.Reals):
            raise ValueError("sets must be a subset of the real line")

        return cls(sets * FiniteSet(0))

    def _contains(self, other):
        from sympy.functions import arg, Abs
        from sympy.core.containers import Tuple
        other = sympify(other)
        isTuple = isinstance(other, Tuple)
        if isTuple and len(other) != 2:
            raise ValueError('expecting Tuple of length 2')

        # If the other is not an Expression, and neither a Tuple
        if not isinstance(other, Expr) and not isinstance(other, Tuple):
            return S.false
        # self in rectangular form
        if not self.polar:
            re, im = other if isTuple else other.as_real_imag()
            for element in self.psets:
                if And(element.args[0]._contains(re),
                        element.args[1]._contains(im)):
                    return True
            return False

        # self in polar form
        elif self.polar:
            if isTuple:
                r, theta = other
            elif other.is_zero:
                r, theta = S.Zero, S.Zero
            else:
                r, theta = Abs(other), arg(other)
            for element in self.psets:
                if And(element.args[0]._contains(r),
                        element.args[1]._contains(theta)):
                    return True
            return False


class Complexes(with_metaclass(Singleton, ComplexRegion)):

    def __new__(cls):
        return ComplexRegion.__new__(cls, S.Reals*S.Reals)

    def __eq__(self, other):
        return other == ComplexRegion(S.Reals*S.Reals)

    def __hash__(self):
        return hash(ComplexRegion(S.Reals*S.Reals))

    def __str__(self):
        return "S.Complexes"

    def __repr__(self):
        return "S.Complexes"
