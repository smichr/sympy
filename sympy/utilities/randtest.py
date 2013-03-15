""" Helpers for randomized testing """

from random import uniform
import random

from sympy import I, nsimplify, Tuple, Symbol
from sympy.core.compatibility import is_sequence, as_int


def random_complex_number(a=2, b=-1, c=3, d=1, rational=False):
    """
    Return a random complex number.

    To reduce chance of hitting branch cuts or anything, we guarantee
    b <= Im z <= d, a <= Re z <= c
    """
    A, B = uniform(a, c), uniform(b, d)
    if not rational:
        return A + I*B
    return nsimplify(A, rational=True) + I*nsimplify(B, rational=True)


def comp(z1, z2, tol):
    """Return a bool indicating whether the error between z1 and z2 is <= tol.

    If z2 is non-zero and ``|z1| > 1`` the error is normalized by ``|z1|``, so
    if you want the absolute error, call this as ``comp(z1 - z2, 0, tol)``.
    """
    if not z1:
        z1, z2 = z2, z1
    if not z1:
        return True
    diff = abs(z1 - z2)
    az1 = abs(z1)
    if z2 and az1 > 1:
        return diff/az1 <= tol
    else:
        return diff <= tol


def test_numerically(f, g, z=None, tol=1.0e-6, a=2, b=-1, c=3, d=1):
    """
    Test numerically that f and g agree when evaluated in the argument z.

    If z is None, all symbols will be tested. This routine does not test
    whether there are Floats present with precision higher than 15 digits
    so if there are, your results may not be what you expect due to round-
    off errors.

    Examples
    ========

    >>> from sympy import sin, cos, S
    >>> from sympy.abc import x
    >>> from sympy.utilities.randtest import test_numerically as tn
    >>> tn(sin(x)**2 + cos(x)**2, 1, x)
    True
    """
    f, g, z = Tuple(f, g, z)
    z = [z] if isinstance(z, Symbol) else (f.free_symbols | g.free_symbols)
    reps = zip(z, [random_complex_number(a, b, c, d) for zi in z])
    z1 = f.subs(reps).n()
    z2 = g.subs(reps).n()
    return comp(z1, z2, tol)


def test_derivative_numerically(f, z, tol=1.0e-6, a=2, b=-1, c=3, d=1):
    """
    Test numerically that the symbolically computed derivative of f
    with respect to z is correct.

    This routine does not test whether there are Floats present with
    precision higher than 15 digits so if there are, your results may
    not be what you expect due to round-off errors.

    Examples
    ========

    >>> from sympy import sin, cos
    >>> from sympy.abc import x
    >>> from sympy.utilities.randtest import test_derivative_numerically as td
    >>> td(sin(x), x)
    True
    """
    from sympy.core.function import Derivative
    z0 = random_complex_number(a, b, c, d)
    f1 = f.diff(z).subs(z, z0)
    f2 = Derivative(f, z).doit_numerically(z0)
    return comp(f1.n(), f2.n(), tol)

import random


def _randrange(seed=None):
    """Return a randrange generator. ``seed`` can be
        o None - return randomly seeded generator
        o int - return a generator seeded with the int
        o list - the values to be returned will be taken from the list
          in the order given; the provided list is not modified.

    Examples
    ========

    >>> from sympy.utilities.randtest import _randrange
    >>> rr = _randrange()
    >>> rr(1000) # doctest: +SKIP
    999
    >>> rr = _randrange(3)
    >>> rr(1000) # doctest: +SKIP
    238
    >>> rr = _randrange([0, 5, 1, 3, 4])
    >>> rr(3), rr(3)
    (0, 1)
    """
    if seed is None:
        return random.randrange
    elif isinstance(seed, int):
        return random.Random(seed).randrange
    elif is_sequence(seed):
        seed = list(seed)  # make a copy
        seed.reverse()

        def give(a, b=None, seq=seed):
            if b is None:
                a, b = 0, a
            a, b = as_int(a), as_int(b)
            w = b - a
            if w < 1:
                raise ValueError('_randrange got empty range')
            try:
                x = seq.pop()
            except AttributeError:
                raise ValueError('_randrange expects a list-like sequence')
            except IndexError:
                raise ValueError('_randrange sequence was too short')
            if a <= x < b:
                return x
            else:
                return give(a, b, seq)
        return give
    else:
        raise ValueError('_randrange got an unexpected seed')


def _randint(seed=None):
    """Return a randint generator. ``seed`` can be
        o None - return randomly seeded generator
        o int - return a generator seeded with the int
        o list - the values to be returned will be taken from the list
          in the order given; the provided list is not modified.

    Examples
    ========

    >>> from sympy.utilities.randtest import _randint
    >>> ri = _randint()
    >>> ri(1, 1000) # doctest: +SKIP
    999
    >>> ri = _randint(3)
    >>> ri(1, 1000) # doctest: +SKIP
    238
    >>> ri = _randint([0, 5, 1, 2, 4])
    >>> ri(1, 3), ri(1, 3)
    (1, 2)
    """
    if seed is None:
        return random.randint
    elif isinstance(seed, int):
        return random.Random(seed).randint
    elif is_sequence(seed):
        seed = list(seed)  # make a copy
        seed.reverse()

        def give(a, b, seq=seed):
            a, b = as_int(a), as_int(b)
            w = b - a
            if w < 0:
                raise ValueError('_randint got empty range')
            try:
                x = seq.pop()
            except AttributeError:
                raise ValueError('_randint expects a list-like sequence')
            except IndexError:
                raise ValueError('_randint sequence was too short')
            if a <= x <= b:
                return x
            else:
                return give(a, b, seq)
        return give
    else:
        raise ValueError('_randint got an unexpected seed')


def symbolics(e, s=None):
    from sympy import Symbol, Dummy, Basic
    if s is None:
        s = set()
    if isinstance(e, (list, tuple)):
        for ei in e:
            symbolics(ei, s)
    if isinstance(e, Symbol):
        s.add(e)
    elif not isinstance(e, Basic):
        pass
    elif e.has(Symbol, Dummy):
        s.add(e)
        for ei in e.args:
            symbolics(ei, s)
    return s


def structurally_equal(a, b):
    m = symbol_match(a, b)
    return m is not None and a.xreplace(m) == b


def symbol_match(a, b, d=None):
    from collections import defaultdict
    from sympy import Basic, Symbol, Dummy, flatten, ordered, Atom
    from sympy.core.compatibility import iterable, set_union, permutations

    if d is None:
        d = {}

    if all(isinstance(i, Symbol) for i in (a, b)):
        if a in d and d[a] != b:
            return
        d[a] = b
        return d

    if iterable(a):
        a = Tuple(*flatten(a))
        b = Tuple(*flatten(b))
        if len(a) != len(b):
            return

    if a == b:
        s = a.atoms(Symbol)  # gets Symbol and Dummy
        d.update(dict(zip(s, s)))
        return d

    if a.func != b.func:
        return

    if len(a.args) != len(b.args):
        return

    def key(i):
        if isinstance(i, (Symbol, Dummy)):
            return None
        return (
            type(i),
            len(i.args),
            tuple(set([(len(i.args), None if isinstance(
                i, (Symbol, Dummy)) else type(i)) for i in i.args])))

    ta = defaultdict(set)
    tb = defaultdict(set)
    for i in a.args:
        ta[key(i)].add(i)
    for i in b.args:
        tb[key(i)].add(i)

    if len(ta) != len(tb):
        return
    if set(ta) != set(tb):
        return

    process = []
    for k in ta.keys():
        if len(ta[k]) != len(tb[k]):
            return
        if len(ta[k]) == 1:
            aa, bb = ta[k].pop(), tb[k].pop()
            rv = symbol_match(aa, bb, d)
            if rv is None:
                return
            del ta[k]
            del tb[k]
        else:
            if k is None:
                process.append((ta.pop(None), tb.pop(None)))
                continue

            def key(i):
                return len(i.atoms(Symbol, Dummy)), i.count_ops(visual=True)

            na = defaultdict(set)
            nb = defaultdict(set)
            for v in ta[k]:
                na[key(v)].add(v)
            for v in tb[k]:
                nb[key(v)].add(v)

            if len(na) != len(nb) or set(na) != set(nb):
                return

            for n in na.keys():
                if len(na[n]) != len(nb[n]):
                    return
                if len(na[n]) == 1:
                    aa, bb = na[n].pop(), nb[n].pop()
                    rv = symbol_match(aa, bb, d)
                    if rv is None:
                        return
                    del na[n]
                    del nb[n]

            for n in na:
                # resolve below
                process.append((na[n], nb[n]))

    if not process:
        return d

    for i, (a, b) in enumerate(process):
        a = [a for a in a if a.has(Symbol, Dummy)]
        b = [b for b in b if b.has(Symbol, Dummy)]
        # filter out those expressions which have are now
        # resolved on the basis of what has been seen
        while True:
            gota = []
            for ai in a:
                if ai.is_Symbol:
                    break
                s = ai.atoms(Symbol, Dummy) - set(d)
                if not s:
                    b.remove(ai.xreplace(d))
                    gota.append(ai)
                else:
                    for bi in b:
                        if symbol_match(ai, bi, d):
                            gota.append(ai)
                            b.remove(ai.xreplace(d))
                            break
            if gota:
                a = [ai for ai in a if ai not in gota]
            else:
                break

        process[i] = (a, b)

    process = [(a,b) for a, b in process if a]

    if not process:
        return d

    for i, (a, b) in enumerate(process):
        a0 = a.pop()
        if isinstance(a0, Symbol):
            # remove known symbols
            newa = []
            for ai in a:
                if ai not in d:
                    newa.append(ai)
                else:
                    b.remove(d[ai])
            a = newa
            if len(a) == 1 == len(b):
                d[a.pop()] = b.pop()
            else:
                process[i] = a, b
        else:
            a.add(a0)

    # simple case (and single symbols don't unzip and flatten well
    if 0 and len(process) == 1 and len(process[0][0]) == 1:
        d[process[0][0].pop()] = process[0][1].pop()
        return d

    # now look for a permutation that will satisfy all expressions
    a, b = [flatten(i) for i in zip(*process)]
    fa = list(ordered(set_union(*[i.free_symbols for i in a])))
    fb = list(ordered(set_union(*[i.free_symbols for i in b])))
    newa = []
    for i in fa:
        if i in d:
            newa.append((i, d[i]))
    if newa:
        for i, j in newa:
            fa.remove(i)
            fb.remove(j)
    for p in permutations(fb):
        reps = dict(zip(fa, p))
        reps.update(d)
        if all(ai.xreplace(reps) in b for ai in a):
            return reps
    return
