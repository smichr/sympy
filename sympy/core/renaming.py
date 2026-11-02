"""Compare nested structures modulo a consistent bijection of SymPy Symbols."""

from collections import Counter, defaultdict
from collections.abc import Mapping

from sympy import Add, FiniteSet, Mul, Symbol
from sympy.core.basic import Basic
from sympy.matrices import MatrixBase
from sympy.core.function import Subs


def same_up_to_renaming(a, b, *, return_mapping=False, explain=False):
    """Test structural equality modulo a bijective renaming of Symbols.

    Handles SymPy Basic objects, matrices, Python mappings, sets, tuples,
    and lists. Add, FiniteSet, and commutative Mul are unordered.
    Other arguments are ordered. Free Symbol types and assumptions are preserved.
    Bound symbols are canonicalized lazily with ``as_dummy()`` when their
    binding node is reached. This separates bound occurrences from free ones.
    Forced comparisons are processed first; unresolved unordered groups are
    considered globally using the smallest candidate group first. Only ties
    surviving structural and occurrence refinement require backtracking.

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
    bound_cache = {}
    scope_counter = 0

    def is_binder(x):
        # Subs.as_dummy does not separate its bound variable from the
        # evaluation point; retain strict matching for Subs for now.
        return (isinstance(x, Basic) and not isinstance(x, Subs)
                and bool(getattr(x, "bound_symbols", ())))

    def canonical_bound(x):
        """Normalize only binders, never bare symbols (whose as_dummy()
        would otherwise erase assumptions).

        This normalization takes place on demand, not as an eager walk over
        the entire input, retaining short-circuiting for ordered containers.
        """
        if not isinstance(x, Basic) or isinstance(x, Symbol):
            return x
        cached = bound_cache.get(id(x))
        if cached is not None and cached[0] is x:
            return cached[1]
        if is_binder(x):  # Integral, Sum, Lambda, ConditionSet, ...
            y = x.as_dummy()
            bound_cache[id(x)] = (x, y)
            return y
        return x

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

    shallow_cache = {}

    def shallow(x, depth=0, side=None, scope=None):
        """Inspect node type and arity first, then children only if needed.

        Unbound Symbols are interchangeable; already bound Symbols carry
        the same mapping color on both sides. A disagreement of keys proves
        that no permissible renaming exists.
        """
        if scope is None:
            scope = {}
        if isinstance(x, Symbol):
            if x in scope:
                return ('scoped-symbol', scope[x])
            fixed = (x if side == 'left' and x in forward else
                     reverse[x] if side == 'right' and x in reverse else None)
            if fixed is not None:
                return ('bound-symbol', fixed)
        cache = shallow_cache if not scope and (side is None or not forward) else None
        if cache is not None:
            item = cache.get((id(x), depth))
            if item is not None and item[0] is x:
                return item[1]
        if depth and is_binder(x):
            x = canonical_bound(x)
            scope = scope.copy()
            for i, symbol in enumerate(x.bound_symbols):
                if isinstance(symbol, Symbol):
                    scope[symbol] = ('local', i)
        tag, xs, unordered = children(x)
        key = (tag, len(xs))
        if depth and xs:
            subkeys = [shallow(t, depth - 1, side, scope) for t in xs]
            key += ((frozenset(Counter(subkeys).items()) if unordered
                     else tuple(subkeys)),)
        if cache is not None:
            cache[id(x), depth] = (x, key)
        return key

    def roles(root):
        """Each symbol's multiset of structural occurrence contexts."""
        occurrences = defaultdict(Counter)

        class HasBinder(Exception):
            pass

        def walk(x, context):
            if is_binder(x):
                raise HasBinder
            if isinstance(x, Symbol):
                occurrences[x][context] += 1
                return
            _, xs, unordered = children(x)
            if not xs:
                return
            parent = shape(x)
            for i, child in enumerate(xs):
                walk(child, context + ((parent, None if unordered else i),))

        try:
            walk(root, ())
        except HasBinder:
            return None
        return {s: (shape(s), frozenset(counts.items()))
                for s, counts in occurrences.items()}

    # Comparisons are propagated first. Ambiguous unordered groups are
    # deferred until every available forced comparison has been processed;
    # then the smallest unresolved group is selected globally (MRV).
    # This also lets later failures revisit earlier unordered choices.
    work = [('compare', a, b, (), ({}, {}))]
    deferred = []
    choices = []
    roles_checked = False
    sifted_binding_count = 0

    def push_pair(x, y, where, scopes):
        work.append(('compare', x, y, where, scopes))

    def push_ordered(xs, ys, where, scopes):
        if len(xs) != len(ys):
            return fail('different lengths', xs, ys, where)
        for i in reversed(range(len(xs))):
            push_pair(xs[i], ys[i], where + (('index', i),), scopes)
        return True

    def push_unordered(xs, ys, where, scopes):
        if len(xs) != len(ys):
            return fail('different lengths', xs, ys, where)
        work.append(('unordered', list(enumerate(xs)),
                     list(enumerate(ys)), where, scopes))
        return True

    def choose(cp, j):
        """Restore a choicepoint, then try its j-th candidate pairing."""
        nonlocal roles_checked, sifted_binding_count
        undo(cp['mark'])
        work[:] = cp['pending']
        deferred[:] = cp['deferred']
        roles_checked = cp['roles_checked']
        sifted_binding_count = cp['sifted_binding_count']
        i, x = cp['first']
        q, y = cp['candidates'][j]
        left = [item for item in cp['left'] if item[0] != i]
        right = [item for item in cp['right'] if item[0] != q]
        if left:
            work.append(('unordered', left, right, cp['where'], cp['scopes']))
        push_pair(x, y, cp['where'] + (('unordered', i, q),), cp['scopes'])

    def process():
        nonlocal roles_checked, sifted_binding_count, scope_counter
        while True:
            good = True
            if not work:
                if not deferred:
                    return True

                # A newly bound symbol may split *any* deferred group into
                # forced correspondences. Re-sift before choosing a branch.
                if len(bindings) != sifted_binding_count:
                    sifted_binding_count = len(bindings)
                    for task in deferred:
                        work.append(('unordered', task['left'],
                                     task['right'], task['where'], task['scopes']))
                    deferred.clear()
                    continue

                if not roles_checked:
                    # Full incidence analysis is a last-resort sieve: do it
                    # only after all shallow and forced work is exhausted.
                    roles_checked = True
                    lr, rr = defaultdict(list), defaultdict(list)
                    left_roles = roles(a)
                    right_roles = roles(b)
                    if left_roles is None or right_roles is None:
                        # Scope-sensitive incidence analysis is not yet
                        # available; search remains complete without it.
                        continue
                    for sym, key in left_roles.items():
                        lr[key].append(sym)
                    for sym, key in right_roles.items():
                        rr[key].append(sym)
                    if (lr.keys() != rr.keys() or
                            any(len(g) != len(rr[k]) for k, g in lr.items())):
                        good = fail('different symbol occurrence patterns', a, b)
                    else:
                        for key, group in lr.items():
                            if len(group) == 1:
                                u, v = group[0], rr[key][0]
                                if u in forward and forward[u] != v:
                                    good = fail('forward conflict', u, v,
                                                previously=forward[u])
                                    break
                                if v in reverse and reverse[v] != u:
                                    good = fail('reverse conflict', u, v,
                                                previously=reverse[v])
                                    break
                                if u not in forward:
                                    bind(u, v)
                    if good:
                        # Any new bindings force another round of sifting.
                        continue
                else:
                    # All forced work and cheap refinements are exhausted.
                    # Choose the smallest unresolved candidate class across
                    # ALL outstanding unordered structures, not just within
                    # the first unordered structure visited.
                    index = min(range(len(deferred)),
                                key=lambda i: deferred[i]['size'])
                    task = deferred.pop(index)
                    cp = {
                        'mark': len(bindings),
                        'pending': work.copy(),
                        'deferred': deferred.copy(),
                        'sifted_binding_count': sifted_binding_count,
                        'roles_checked': roles_checked,
                        'scopes': task['scopes'],
                        'left': task['left'], 'right': task['right'],
                        'first': task['first'],
                        'candidates': task['candidates'], 'next': 1,
                        'where': task['where'],
                    }
                    choices.append(cp)
                    choose(cp, 0)
                    continue

            else:
                kind, x, y, where, scopes = work.pop()

                if kind == 'unordered':
                    left, right = x, y
                    if not left:
                        continue
                    # Refine only ambiguous classes. Stop as soon as a
                    # forced pairing appears; it may resolve larger groups.
                    for depth in range(3):
                        lg, rg = defaultdict(list), defaultdict(list)
                        for i, a0 in left:
                            lg[shallow(a0, depth, 'left', scopes[0])].append((i, a0))
                        for j, b0 in right:
                            rg[shallow(b0, depth, 'right', scopes[1])].append((j, b0))
                        if lg.keys() != rg.keys() or any(
                            len(v) != len(rg[k]) for k, v in lg.items()
                        ):
                            good = fail('incompatible unordered groups',
                                        [v for _, v in left],
                                        [v for _, v in right], where,
                                        refinement_depth=depth)
                            break
                        forced = [(group[0], rg[k][0])
                                  for k, group in lg.items() if len(group) == 1]
                        if forced:
                            break
                    if good:
                        if forced:
                            forced_left = {p[0][0] for p in forced}
                            forced_right = {p[1][0] for p in forced}
                            remaining_left = [p for p in left
                                              if p[0] not in forced_left]
                            remaining_right = [p for p in right
                                               if p[0] not in forced_right]
                            if remaining_left:
                                work.append(('unordered', remaining_left,
                                             remaining_right, where, scopes))
                            for (i, a0), (j, b0) in reversed(forced):
                                push_pair(a0, b0,
                                          where + (('unordered', i, j),), scopes)
                        else:
                            key = min(lg, key=lambda k: len(lg[k]))
                            deferred.append({
                                'left': left, 'right': right, 'where': where,
                                'first': lg[key][0],
                                'candidates': rg[key], 'size': len(lg[key]),
                                'scopes': scopes,
                            })
                    if good:
                        continue

                elif isinstance(x, Symbol) or isinstance(y, Symbol):
                    xbound = scopes[0].get(x) if isinstance(x, Symbol) else None
                    ybound = scopes[1].get(y) if isinstance(y, Symbol) else None
                    if xbound is not None or ybound is not None:
                        if xbound is None or xbound != ybound:
                            good = fail('different bound variables', x, y, where)
                    elif type(x) is not type(y):
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
                        good = push_ordered(list(x), list(y), where, scopes)

                elif isinstance(x, Mapping) or isinstance(y, Mapping):
                    if not isinstance(x, Mapping) or not isinstance(y, Mapping):
                        good = fail('different types', x, y, where)
                    else:
                        good = push_unordered(list(x.items()), list(y.items()), where, scopes)

                elif isinstance(x, (set, frozenset)) or isinstance(y, (set, frozenset)):
                    if type(x) is not type(y):
                        good = fail('different set types', x, y, where)
                    else:
                        good = push_unordered(list(x), list(y), where, scopes)

                elif isinstance(x, Basic) or isinstance(y, Basic):
                    if type(x) is not type(y):
                        good = fail('different SymPy types', x, y, where)
                    elif is_binder(x) or is_binder(y):
                        if not (is_binder(x) and is_binder(y)):
                            good = fail('different binding structures', x, y, where)
                        else:
                            ux, uy = canonical_bound(x), canonical_bound(y)
                            bx, by = ux.bound_symbols, uy.bound_symbols
                            if len(bx) != len(by):
                                good = fail('different binding arities', x, y, where)
                            else:
                                scope_counter += 1
                                left_scope, right_scope = (dict(scopes[0]),
                                                           dict(scopes[1]))
                                for i, (lx, ry) in enumerate(zip(bx, by)):
                                    token = (scope_counter, i)
                                    if isinstance(lx, Symbol):
                                        left_scope[lx] = token
                                    if isinstance(ry, Symbol):
                                        right_scope[ry] = token
                                good = push_ordered(ux.args, uy.args, where,
                                                    (left_scope, right_scope))
                    elif not x.args:
                        if x != y:
                            good = fail('different atoms', x, y, where)
                    elif isinstance(x, (Add, FiniteSet)) or (
                        isinstance(x, Mul) and x.is_commutative
                    ):
                        good = push_unordered(x.args, y.args, where, scopes)
                    else:
                        good = push_ordered(x.args, y.args, where, scopes)

                elif isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
                    if type(x) is not type(y):
                        good = fail('different sequence types', x, y, where)
                    else:
                        good = push_ordered(x, y, where, scopes)
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

    # Ordered comparisons proceed left-to-right and stop at the first
    # disagreement. Unordered groups inspect structure only as necessary.
    result = process()
    mapping = forward.copy() if result else None
    if not result:
        undo(0)
    if explain:
        return result, mapping, None if result else best_error
    return mapping if return_mapping else result
