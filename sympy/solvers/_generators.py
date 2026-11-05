"""Generator substitutions shared by solve and nonlinsolve.

This module only changes variables. Solving the lifted equations and
choosing an inverse are left to the respective solvers.
"""

from __future__ import annotations

from itertools import product
from typing import NamedTuple

from sympy.core.expr import Expr
from sympy.core.exprtools import _decompose_exprs
from sympy.core.relational import Eq
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.core.traversal import preorder_traversal
from sympy.functions.elementary.exponential import exp


class GeneratorLift(NamedTuple):
    equations: tuple
    symbols: tuple
    original_symbols: tuple
    substitutions: tuple


def _lift_generators(equations, symbols):
    """Replace suitable univariate exponential generators with symbols.

    Return None if a polynomial/rational lift cannot be made safely.
    Denominators and incomplete substitutions are left to the usual solver.
    """
    exprs = [
        e.lhs - e.rhs if isinstance(e, Eq) else e
        for e in equations
    ]
    if not all(isinstance(e, Expr) and e.is_commutative for e in exprs):
        return None

    original_symbols = tuple(symbols)
    symset = set(original_symbols)
    # Do not lose domain restrictions in the original equations.
    if any(e.as_numer_denom()[1].free_symbols & symset for e in exprs):
        return None

    # Include nested sums in otherwise unexpanded products.
    parts = exprs + [
        a for e in exprs for a in preorder_traversal(e) if a.is_Add
    ]
    _, gens = _decompose_exprs(parts)

    lifted = exprs
    new_symbols = []
    substitutions = []
    for s in original_symbols:
        candidates = sorted(
            (g for g in gens if g.func is exp and g.free_symbols == {s}),
            key=default_sort_key)
        for g in candidates:
            t = Dummy('g')
            trial = [e.subs(g, t) for e in lifted]
            if all(s not in e.free_symbols for e in trial):
                lifted = trial
                new_symbols.append(t)
                substitutions.append((s, g, t))
                break
        else:
            new_symbols.append(s)

    if not substitutions:
        return None
    if not all(
        all(p.is_polynomial(*new_symbols) for p in e.as_numer_denom())
        for e in lifted
    ):
        return None

    return GeneratorLift(
        tuple(lifted), tuple(new_symbols), original_symbols,
        tuple(substitutions))


def _recover_generators(lift, solutions, invert):
    """Convert lifted solution dictionaries to original variables.

    `invert(generator, value, symbol)` returns an iterable of inverse
    values, an empty iterable for no inverse, or None on failure.
    Return None if any branch cannot be recovered safely.
    """
    inverse_data = {s: (g, t) for s, g, t in lift.substitutions}
    new_symbols = set(lift.symbols)
    recovered = []

    for sol in solutions:
        if any(s not in sol for s in lift.symbols):
            return None
        # For now, defer positive-dimensional and set-valued lifted
        # systems. The inversions below assume fixed generator values.
        if any(not isinstance(v, Expr) or v.free_symbols & new_symbols
               for v in sol.values()):
            return None

        choices = []
        for old, new in zip(lift.original_symbols, lift.symbols):
            value = sol[new]
            if old in inverse_data:
                g, _ = inverse_data[old]
                if value.free_symbols and value.is_zero is not False:
                    return None
                values = invert(g, value, old)
                if values is None:
                    return None
                if not values:
                    break
                choices.append(tuple(values))
            else:
                choices.append((value,))
        else:
            recovered.extend(
                dict(zip(lift.original_symbols, values))
                for values in product(*choices))

    return recovered
