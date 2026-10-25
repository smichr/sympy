from __future__ import annotations
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.operations import AssocOp, LatticeOp
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
from sympy.core.add import Add, add
from sympy.core.mul import Mul, mul

# create the simplest possible Lattice class


class join(LatticeOp):
    zero = Integer(0)
    identity = Integer(1)


def test_lattice_simple():
    assert join(join(2, 3), 4) == join(2, join(3, 4))
    assert join(2, 3) == join(3, 2)
    assert join(0, 2) == 0
    assert join(1, 2) == 2
    assert join(2, 2) == 2

    assert join(join(2, 3), 4) == join(2, 3, 4)
    assert join() == 1
    assert join(4) == 4
    assert join(1, 4, 2, 3, 1, 3, 2) == join(2, 3, 4)


def test_lattice_shortcircuit():
    raises(SympifyError, lambda: join(object))
    assert join(0, object) == 0


def test_lattice_print():
    assert str(join(5, 4, 3, 2)) == 'join(2, 3, 4, 5)'


def test_lattice_make_args():
    assert join.make_args(join(2, 3, 4)) == {S(2), S(3), S(4)}
    assert join.make_args(0) == {0}
    assert list(join.make_args(0))[0] is S.Zero
    assert Add.make_args(0)[0] is S.Zero


def test_issue_14025():
    a, b, c, d = symbols('a,b,c,d', commutative=False)
    assert Mul(a, b, c).has(c*b) == False
    assert Mul(a, b, c).has(b*c) == True
    assert Mul(a, b, c, d).has(b*c*d) == True


def test_AssocOp_flatten():
    a, b, c, d = symbols('a,b,c,d')

    class MyAssoc(AssocOp):
        identity = S.One

    assert MyAssoc(a, MyAssoc(b, c)).args == \
        MyAssoc(MyAssoc(a, b), c).args == \
        MyAssoc(MyAssoc(a, b, c)).args == \
        MyAssoc(a, b, c).args == \
            (a, b, c)
    u = MyAssoc(b, c)
    v = MyAssoc(u, d, evaluate=False)
    assert v.args == (u, d)
    # like Add, any unevaluated outer call will flatten inner args
    assert MyAssoc(a, v).args == (a, b, c, d)


def test_add_dispatcher():

    class NewBase(Expr):
        @property
        def _add_handler(self):
            return NewAdd
    class NewAdd(NewBase, Add):
        pass
    add.register_handlerclass((Add, NewAdd), NewAdd)

    a, b = Symbol('a'), NewBase()

    # Add called as fallback
    assert add(1, 2) == Add(1, 2)
    assert add(a, a) == Add(a, a)

    # selection by registered priority
    assert add(a,b,a) == NewAdd(2*a, b)


def test_mul_dispatcher():

    class NewBase(Expr):
        @property
        def _mul_handler(self):
            return NewMul
    class NewMul(NewBase, Mul):
        pass
    mul.register_handlerclass((Mul, NewMul), NewMul)

    a, b = Symbol('a'), NewBase()

    # Mul called as fallback
    assert mul(1, 2) == Mul(1, 2)
    assert mul(a, a) == Mul(a, a)

    # selection by registered priority
    assert mul(a,b,a) == NewMul(a**2, b)


def test_unevaluate_assocop():
    from sympy import evaluate
    from collections import Counter

    x, y, z = symbols('x y z')
    A, B = symbols('A B', commutative=False)

    for op, args in [
        (Add, (x, y, z)), (Add, (x, x, y)), (Add, (S.One, S(2), x)),
        (Mul, (x, y, z)), (Mul, (x, x, y)), (Mul, (S(2), S(3), x)),
    ]:
        f = op(*args, unevaluate=True)
        g = op(*reversed(args), unevaluate=True)
        assert f == g
        assert f.args == g.args
        assert Counter(f.args) == Counter(args)
        assert f.func is op
        nested = op(args[0], op(*args[1:], evaluate=False), unevaluate=True)
        assert nested == f
        assert nested.args == f.args

    # Literal ordering is unchanged.
    assert Mul(y, x, evaluate=False).args == (y, x)
    assert Add(y, x, evaluate=False).args == (y, x)

    # Flattening removes associative groupings, not operands.
    assert Mul(x, Mul(y, z, evaluate=False), unevaluate=True) == Mul(z, y, x, unevaluate=True)
    assert Add(x, Add(y, z, evaluate=False), unevaluate=True) == Add(z, y, x, unevaluate=True)

    # These are objects, not the evaluated results.
    assert Add(x, x, unevaluate=True).args == (x, x)
    assert Mul(x, x, unevaluate=True).args == (x, x)
    assert Add(S(2), S(3), unevaluate=True).args == (S(2), S(3))
    assert Mul(S(2), S(3), unevaluate=True).args == (S(2), S(3))
    assert Add(x, S.Zero, unevaluate=True).is_Add
    assert Mul(x, S.One, unevaluate=True).is_Mul
    # No automatic distribution, even when an ordinary Add is provided.
    assert Mul(S(2), Add(x, S.One), unevaluate=True).is_Mul
    assert Mul(A, B, unevaluate=True) != Mul(B, A, unevaluate=True)
    assert Mul(y, A, x, B, unevaluate=True) == Mul(x, A, y, B, unevaluate=True)

    with evaluate(False):
        assert Add(y, x, unevaluate=True) == Add(x, y, unevaluate=True)

    raises(ValueError, lambda: Add(x, y, evaluate=False, unevaluate=True))
    raises(ValueError, lambda: Mul(x, y, evaluate=True, unevaluate=True))


def test_post_process_with_unevaluate():
    from sympy import MatAdd, MatMul, MatrixSymbol

    A = MatrixSymbol('A', 2, 2)

    a = Add(A, A, unevaluate=True)
    assert isinstance(a, MatAdd)
    assert a.args == (A, A)

    m = Mul(A, A, unevaluate=True)
    assert isinstance(m, MatMul)
    assert m.args == (A, A)
