from collections.abc import Mapping
from itertools import combinations, permutations
from random import Random
from sympy import Symbol, symbols, Add, Mul, sin, cos, Eq, FiniteSet, Matrix, Integer
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.matrices import MatrixBase
from sympy.core.renaming import same_up_to_renaming


def test_bijective_renaming():
    def baseline(a, b, *, return_mapping=False):
        forward, reverse = {}, {}
        def ordered(xs, ys):
            return len(xs) == len(ys) and all(match(x, y) for x, y in zip(xs, ys))
        def unordered(xs, ys):
            if len(xs) != len(ys): return False
            if not xs: return True
            for i, y in enumerate(ys):
                old_f, old_r = forward.copy(), reverse.copy()
                if match(xs[0], y) and unordered(xs[1:], ys[:i] + ys[i + 1:]):
                    return True
                forward.clear(); forward.update(old_f)
                reverse.clear(); reverse.update(old_r)
            return False
        def match(x, y):
            if isinstance(x, Symbol) or isinstance(y, Symbol):
                if type(x) is not type(y) or x.assumptions0 != y.assumptions0: return False
                if x in forward: return forward[x] == y
                if y in reverse: return False
                forward[x] = y; reverse[y] = x; return True
            if isinstance(x, MatrixBase) or isinstance(y, MatrixBase):
                return isinstance(x, MatrixBase) and isinstance(y, MatrixBase) and x.shape == y.shape and ordered(list(x), list(y))
            if isinstance(x, Mapping) or isinstance(y, Mapping):
                return isinstance(x, Mapping) and isinstance(y, Mapping) and unordered(list(x.items()), list(y.items()))
            if isinstance(x, (set, frozenset)) or isinstance(y, (set, frozenset)):
                return type(x) is type(y) and unordered(list(x), list(y))
            if isinstance(x, Basic) or isinstance(y, Basic):
                if type(x) is not type(y): return False
                if not x.args: return x == y
                if isinstance(x, (Add, FiniteSet)) or isinstance(x, Mul) and x.is_commutative:
                    return unordered(list(x.args), list(y.args))
                return ordered(x.args, y.args)
            if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
                return type(x) is type(y) and ordered(x, y)
            return type(x) is type(y) and x == y
        result = match(a, b)
        return (forward.copy() if result else None) if return_mapping else result

    x,y,z,w = symbols('x y z w')
    a,b,c,d = symbols('a b c d')

    examples = [
      (x+2*y+3*z, 2*a+b+3*c, True),
      ([x,x], [a,b], False),
      ([x,y], [a,a], False),
      ([x,y,x], [a,b,a], True),
      ([x,y,x], [a,b,b], False),
      ({x:y,y:x}, {b:a,a:b}, True),
      (set((x,y)),set((a,b)),True),
      ({x, y}, {a, Integer(2)},False),
      (Matrix([[x,y],[x,3]]),Matrix([[a,b],[a,3]]),True),
      (sin(x+y),sin(a+b),True),
      (sin(x+x),sin(a+b),False),
      (FiniteSet(x,y),FiniteSet(a,b),True),
      (Symbol('t',real=True),Symbol('u'),False),
      (x*y+y*z+z*w+w*x, a*b+b*c+c*d+d*a, True),
      (x*y + y*z + z*w + w*x, a*b + b*c + c*d + d*b, False),
    ]
    for left,right,expect in examples:
        got=same_up_to_renaming(left,right)
        old=baseline(left,right)
        assert got==expect,(left,right,got,expect)
        assert old==expect,(left,right,old,expect)

    # Regression: the original locally backtracks within FiniteSet but cannot
    # reconsider its locally successful pairing when the outer tuple later fails.
    left = (FiniteSet(x, y), x)
    right = (FiniteSet(a, b), b)
    assert baseline(left, right) is False
    assert same_up_to_renaming(left, right, return_mapping=True) == {x: b, y: a}

    rnd=Random(12345)
    ls=[x,y,z,w];rs=[a,b,c,d]
    def expression(depth, alphabet):
        if not depth or rnd.randrange(4)==0:
            return rnd.choice(alphabet+[Integer(1),Integer(2),Integer(3)])
        args=[expression(depth-1,alphabet) for _ in range(rnd.choice((2,2,3)))]
        operation=rnd.choice(['add','mul','sin','cos','tuple','list','dict','set','matrix'])
        if operation=='add': return Add(*args, evaluate=False) if all(isinstance(v, Expr) for v in args) else tuple(args)
        if operation=='mul': return Mul(*args, evaluate=False) if all(isinstance(v, Expr) for v in args) else tuple(args)
        if operation=='sin': return sin(args[0]) if isinstance(args[0], Expr) else tuple(args)
        if operation=='cos': return cos(args[0]) if isinstance(args[0], Expr) else tuple(args)
        if operation=='tuple': return tuple(args)
        if operation=='list': return list(args)
        if operation=='dict': return {i:arg for i,arg in enumerate(args)}
        if operation=='set': return set(args) if all(isinstance(v, Basic) for v in args) else tuple(args)
        if operation=='matrix': return Matrix([args]) if all(isinstance(v, Expr) for v in args) else list(args)

    for i in range(1300):
        lhs=expression(3,ls)
        rhs=expression(3,rs)
        old=baseline(lhs,rhs)
        got=same_up_to_renaming(lhs,rhs)
        assert old==got,(i,lhs,rhs,old,got)

    # Matched examples under random globally-consistent permutations, including
    # arithmetic expressions whose construction can be re-canonicalized by SymPy.
    for i in range(500):
        lhs=expression(3,ls)
        perm=rnd.sample(rs,4)
        # Avoid sympify on mixed Python types, use tree substitution via visitor
        def remap(v):
            if isinstance(v,Symbol): return perm[ls.index(v)]
            if isinstance(v, MatrixBase): return v.applyfunc(remap)
            if isinstance(v,dict): return {remap(k):remap(q) for k,q in v.items()}
            if isinstance(v,list): return list(map(remap,v))
            if isinstance(v,tuple): return tuple(map(remap,v))
            if isinstance(v,set): return set(map(remap,v))
            if isinstance(v,Basic) and v.args: return v.func(*[remap(q) for q in v.args],evaluate=False) if v.func in (Add,Mul) else v.func(*[remap(q) for q in v.args])
            return v
        rhs=remap(lhs)
        old=baseline(lhs,rhs)
        got=same_up_to_renaming(lhs,rhs)
        assert got,(i,lhs,rhs,old,got)

    # Independent exhaustive oracle for small simple undirected graphs:
    # all n! symbol bijections are tried via xreplace and equality.
    vertices_l = symbols('v:5')
    vertices_r = symbols('u:5')
    for t in range(300):
        n = rnd.choice((3, 4, 5))
        vv, uu = vertices_l[:n], vertices_r[:n]
        edges = list(combinations(range(n), 2))
        el = rnd.sample(edges, rnd.randrange(1, len(edges) + 1))
        er = rnd.sample(edges, len(el))
        left = Add(*[vv[i]*vv[j] for i, j in el])
        right = Add(*[uu[i]*uu[j] for i, j in er])
        expected = any(left.xreplace(dict(zip(vv, perm))) == right
                       for perm in permutations(uu))
        assert same_up_to_renaming(left, right) == expected

    if __name__ == '__main__':
        print('success: 15 fixed + nested regression + 1300 random pairs + '
          '500 renamed structures + 300 brute-force graph checks')
        print('example mapping:', same_up_to_renaming(x+2*y+3*z, 2*a+b+3*c, return_mapping=True))
        print('example explain:', same_up_to_renaming([x,y], [a,a], explain=True))
