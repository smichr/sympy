"""Compare nested structures modulo a consistent bijection of SymPy Symbols."""

from collections import Counter, defaultdict
from collections.abc import Mapping

from sympy import Add, FiniteSet, Mul, Symbol
from sympy.core.basic import Basic
from sympy.matrices import MatrixBase


def same_up_to_renaming(a, b, *, return_mapping=False, explain=False):
    """Test structural equality modulo a bijective renaming of Symbols.

    Handles SymPy Basic objects, matrices, Python mappings, sets, tuples,
    and lists. Add, FiniteSet, and commutative Mul are unordered.
    Other arguments are ordered. Symbol type and assumptions are preserved.

    ``return_mapping=True`` returns the successful mapping, or None.
    ``explain=True`` returns ``(matched, mapping_or_None, error_or_None)``.
    When search is necessary, an error describes one of the most advanced
    failed branches, not necessarily a uniquely determined obstruction.
    """
    forward, reverse = {}, {}
    bindings = []
    best_error = None
    best_score = (-1, -1)
    plain_cache = {}

    def bind(x, y):
        forward[x] = y
        reverse[y] = x
        bindings.append((x, y))

    def undo(mark):
        while len(bindings) > mark:
            x, y = bindings.pop()
            del forward[x]
            del reverse[y]

    def fail(reason, x, y, where=(), **detail):
        nonlocal best_error, best_score
        if explain:
            score = len(bindings), len(where)
            if score > best_score:
                best_score = score
                best_error = dict(reason=reason, path=where,
                                  left=x, right=y, mapping=forward.copy(),
                                  **detail)
        return False

    def children(x):
        """Return (immutable node label, children, unordered?) or None."""
        if isinstance(x, Symbol):
            return ('symbol', type(x), tuple(sorted(x.assumptions0.items()))), (), False
        if isinstance(x, MatrixBase):
            return ('matrix', x.shape), tuple(x), False
        if isinstance(x, Mapping):
            return ('mapping',), tuple(x.items()), True
        if isinstance(x, (set, frozenset)):
            return ('set', type(x)), tuple(x), True
        if isinstance(x, Basic):
            if not x.args:
                return ('basic-atom', type(x), x), (), False
            unordered = isinstance(x, (Add, FiniteSet)) or (
                isinstance(x, Mul) and x.is_commutative)
            return ('basic', type(x)), x.args, unordered
        if isinstance(x, (list, tuple)):
            return ('sequence', type(x)), x, False
        try:
            hash(x)
        except TypeError:
            # A coarse (not unsafe) key. Actual equality is checked by match.
            return ('opaque-value', type(x)), (), False
        return ('value', type(x), x), (), False

    def shape(x, side=None):
        """Exact structural signature, except that symbol names are ignored.

        When side is 'left' or 'right', already-bound symbols carry the
        same identifying color on both sides, allowing further refinement.
        """
        if isinstance(x, Symbol) and side is not None:
            fixed = (x if side == 'left' and x in forward else
                     reverse[x] if side == 'right' and x in reverse else None)
            if fixed is not None:
                return ('bound-symbol', fixed)
        if side is None:
            old = plain_cache.get(id(x))
            if old is not None and old[0] is x:
                return old[1]
        tag, xs, unordered = children(x)
        if xs:
            keys = [shape(arg, side) for arg in xs]
            parts = (frozenset(Counter(keys).items()) if unordered
                     else tuple(keys))
            key = (tag, parts)
        else:
            key = (tag,)
        if side is None:
            plain_cache[id(x)] = (x, key)
        return key

    def roles(root):
        """Each symbol's multiset of structural occurrence contexts."""
        occurrences = defaultdict(Counter)

        def walk(x, context):
            if isinstance(x, Symbol):
                occurrences[x][context] += 1
                return
            _, xs, unordered = children(x)
            if not xs:
                return
            parent = shape(x)
            for i, child in enumerate(xs):
                walk(child, context + ((parent, None if unordered else i),))

        walk(root, ())
        return {s: (shape(s), frozenset(counts.items()))
                for s, counts in occurrences.items()}

    # Pending comparisons and unordered groups. Unlike recursive local
    # backtracking, these tasks let a later mismatch revisit ANY earlier
    # ambiguous pairing, including one nested inside an ordered container.
    work = [('compare', a, b, ())]
    choices = []

    def push_pair(x, y, where):
        work.append(('compare', x, y, where))

    def push_ordered(xs, ys, where):
        if len(xs) != len(ys):
            return fail('different lengths', xs, ys, where)
        for i in reversed(range(len(xs))):
            push_pair(xs[i], ys[i], where + (('index', i),))
        return True

    def push_unordered(xs, ys, where):
        if len(xs) != len(ys):
            return fail('different lengths', xs, ys, where)
        work.append(('unordered', list(enumerate(xs)),
                     list(enumerate(ys)), where))
        return True

    def choose(cp, j):
        """Restore a choicepoint, then try its j-th candidate pairing."""
        undo(cp['mark'])
        work[:] = cp['pending']
        i, x = cp['first']
        q, y = cp['candidates'][j]
        left = [item for item in cp['left'] if item[0] != i]
        right = [item for item in cp['right'] if item[0] != q]
        if left:
            work.append(('unordered', left, right, cp['where']))
        push_pair(x, y, cp['where'] + (('unordered', i, q),))

    def process():
        while True:
            if not work:
                return True
            kind, x, y, where = work.pop()
            good = True

            if kind == 'unordered':
                left, right = x, y
                if not left:
                    continue
                lg, rg = defaultdict(list), defaultdict(list)
                for i, a0 in left:
                    lg[shape(a0, 'left')].append((i, a0))
                for j, b0 in right:
                    rg[shape(b0, 'right')].append((j, b0))
                if lg.keys() != rg.keys() or any(
                    len(v) != len(rg[k]) for k, v in lg.items()
                ):
                    good = fail('incompatible unordered groups',
                                [v for _, v in left],
                                [v for _, v in right], where)
                else:
                    forced, unresolved_left, unresolved_right = [], [], []
                    for k, group in lg.items():
                        others = rg[k]
                        if len(group) == 1:
                            forced.append((group[0], others[0]))
                        else:
                            unresolved_left.extend(group)
                            unresolved_right.extend(others)
                    if forced:
                        if unresolved_left:
                            work.append(('unordered', unresolved_left,
                                         unresolved_right, where))
                        for (i, a0), (j, b0) in reversed(forced):
                            push_pair(a0, b0, where + (('unordered', i, j),))
                    else:
                        k = min(lg, key=lambda k: len(lg[k]))
                        cp = {'mark': len(bindings), 'pending': work.copy(),
                              'left': left, 'right': right,
                              'first': lg[k][0],
                              'candidates': rg[k], 'next': 1,
                              'where': where}
                        choices.append(cp)
                        choose(cp, 0)

            elif isinstance(x, Symbol) or isinstance(y, Symbol):
                if type(x) is not type(y):
                    good = fail('different symbol types', x, y, where)
                elif x.assumptions0 != y.assumptions0:
                    good = fail('different symbol assumptions', x, y, where)
                elif x in forward:
                    if forward[x] != y:
                        good = fail('forward conflict', x, y, where,
                                    previously=forward[x])
                elif y in reverse:
                    good = fail('reverse conflict', x, y, where,
                                previously=reverse[y])
                else:
                    bind(x, y)

            elif isinstance(x, MatrixBase) or isinstance(y, MatrixBase):
                if not isinstance(x, MatrixBase) or not isinstance(y, MatrixBase):
                    good = fail('different types', x, y, where)
                elif x.shape != y.shape:
                    good = fail('different matrix shapes', x, y, where)
                else:
                    good = push_ordered(list(x), list(y), where)

            elif isinstance(x, Mapping) or isinstance(y, Mapping):
                if not isinstance(x, Mapping) or not isinstance(y, Mapping):
                    good = fail('different types', x, y, where)
                else:
                    good = push_unordered(list(x.items()), list(y.items()), where)

            elif isinstance(x, (set, frozenset)) or isinstance(y, (set, frozenset)):
                if type(x) is not type(y):
                    good = fail('different set types', x, y, where)
                else:
                    good = push_unordered(list(x), list(y), where)

            elif isinstance(x, Basic) or isinstance(y, Basic):
                if type(x) is not type(y):
                    good = fail('different SymPy types', x, y, where)
                elif not x.args:
                    if x != y:
                        good = fail('different atoms', x, y, where)
                elif isinstance(x, (Add, FiniteSet)) or (
                    isinstance(x, Mul) and x.is_commutative
                ):
                    good = push_unordered(x.args, y.args, where)
                else:
                    good = push_ordered(x.args, y.args, where)

            elif isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
                if type(x) is not type(y):
                    good = fail('different sequence types', x, y, where)
                else:
                    good = push_ordered(x, y, where)
            elif type(x) is not type(y):
                good = fail('different types', x, y, where)
            elif x != y:
                good = fail('different values', x, y, where)

            if good:
                continue
            # The entire pending worklist is rewound at a choicepoint.
            # This matters if earlier unordered matching "succeeded"
            # before encountering a conflict in an enclosing structure.
            while choices:
                cp = choices[-1]
                j = cp['next']
                if j < len(cp['candidates']):
                    cp['next'] += 1
                    choose(cp, j)
                    break
                choices.pop()
            else:
                return False

    def explain_shape_difference(x, y, where=()):
        """Locate a necessary structural mismatch without searching pairings."""
        if isinstance(x, Symbol) and isinstance(y, Symbol):
            if type(x) is not type(y):
                return fail('different symbol types', x, y, where)
            return fail('different symbol assumptions', x, y, where)
        tx, xs, ux = children(x)
        ty, ys, uy = children(y)
        if tx != ty or ux != uy:
            return fail('different structural types or shapes', x, y, where)
        if len(xs) != len(ys):
            return fail('different lengths', x, y, where)
        if not xs:
            return fail('different atomic values', x, y, where)
        if ux:
            # There is no positional path through an unordered mismatch;
            # identify the enclosing group rather than invent a pairing.
            return fail('incompatible unordered groups', x, y, where)
        for i, (u, v) in enumerate(zip(xs, ys)):
            if shape(u) != shape(v):
                return explain_shape_difference(u, v,
                                                where + (('index', i),))
        return fail('different structural signatures', x, y, where)

    # First reject distinct skeletons, then assign symbols whose structural
    # occurrence patterns force a unique correspondence.
    if shape(a) != shape(b):
        result = (explain_shape_difference(a, b) if explain else False)
    else:
        lr, rr = defaultdict(list), defaultdict(list)
        for s, k in roles(a).items():
            lr[k].append(s)
        for s, k in roles(b).items():
            rr[k].append(s)
        if (lr.keys() != rr.keys() or
                any(len(v) != len(rr[k]) for k, v in lr.items())):
            result = fail('different symbol occurrence patterns', a, b)
        else:
            for k, ss in lr.items():
                if len(ss) == 1:
                    bind(ss[0], rr[k][0])
            result = process()

    mapping = forward.copy() if result else None
    if not result:
        undo(0)
    if explain:
        return result, mapping, None if result else best_error
    return mapping if return_mapping else result
