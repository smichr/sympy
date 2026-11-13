"""Experimental structural arithmetic used by the optional parser mode.

Ordinary unevaluated Mul nodes preserve the order and nesting of written
multiplication. Div retains explicit division distinct from a reciprocal
power. Call touch() to evaluate arithmetic operations inside-out.
"""

from __future__ import annotations
import ast

from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sympify import sympify
from sympy.parsing.sympy_parser import EvaluateFalseTransformer
from sympy.printing.precedence import PRECEDENCE


def _operand_str(printer, arg):
    result = printer._print(arg)
    if (arg.is_Add or arg.is_Mul or isinstance(arg, Div) or
            (arg.is_Rational and arg.q != 1) or
            (arg.is_Pow and arg.exp.is_negative) or
            (arg.is_Number and arg.is_negative)):
        return '(%s)' % result
    return result


class Div(Function):
    """Explicit unevaluated binary division operation."""
    nargs = 2
    # Parenthesize Div when printed as a multiplicative/power operand.
    precedence = PRECEDENCE['Mul'] - 1

    def doit(self, deep=True, **hints):
        a, b = self.args
        if deep:
            a, b = (arg.doit(deep=True, **hints) for arg in (a, b))
        return a/b

    def _sympystr(self, printer):
        a, b = self.args
        return '%s/%s' % (_operand_str(printer, a),
                          _operand_str(printer, b))

    def _latex(self, printer, exp=None):
        a, b = self.args
        result = r'\frac{%s}{%s}' % (printer._print(a), printer._print(b))
        if exp is not None:
            return r'\left(%s\right)^{%s}' % (result, exp)
        return result


def touch(expr):
    """Evaluate arithmetic operations in their original nesting order.

    Unlike a general doit(), this does not explicitly evaluate arbitrary
    functions, integrals, sums, or other non-arithmetic operations.
    """
    return sympify(expr).replace(
        lambda e: isinstance(e, (Add, Mul, Pow, Div)),
        lambda e: e.doit(deep=False) if isinstance(e, Div) else e.func(*e.args))


class StructuralTransformer(EvaluateFalseTransformer):
    """Retain binary multiplication and explicit division in parse_expr's AST."""

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.Mult, ast.Div)):
            name = ('Mul' if isinstance(node.op, ast.Mult)
                    else '_sympy_parsing_div')
            keywords = ([ast.keyword(arg='evaluate', value=ast.Constant(False))]
                        if isinstance(node.op, ast.Mult) else [])
            return ast.Call(
                func=ast.Name(id=name, ctx=ast.Load()),
                args=[self.visit(node.left), self.visit(node.right)],
                keywords=keywords)
        return super().visit_BinOp(node)


def evaluate_structural(s):
    """Return an AST retaining multiply/divide operations without flattening."""
    node = ast.parse(s)
    transformed = StructuralTransformer().visit(node)
    transformed = ast.Expression(transformed.body[0].value)
    return ast.fix_missing_locations(transformed)


def parse_structural_expr(s, local_dict=None, transformations=None,
                          global_dict=None):
    """Experimentally parse multiplication and division without flattening.

    Uses parse_expr's token transformations and its non-evaluating AST
    transformer for other operators. Parentheses are used to determine
    the AST but redundant parentheses are not retained as separate nodes.

    Like parse_expr, this function uses eval and must not be used with
    untrusted input.
    """
    import builtins
    import types

    from sympy.parsing.sympy_parser import (
        T, eval_expr, null, standard_transformations, stringify_expr)

    if local_dict is None:
        local_dict = {}
    elif not isinstance(local_dict, dict):
        raise TypeError('expecting local_dict to be a dict')

    if global_dict is None:
        global_dict = {}
        exec('from sympy import *', global_dict)
        for name, obj in vars(builtins).items():
            if isinstance(obj, types.BuiltinFunctionType):
                global_dict[name] = obj
        from sympy import Max, Min
        global_dict['max'] = Max
        global_dict['min'] = Min
    elif not isinstance(global_dict, dict):
        raise TypeError('expecting global_dict to be a dict')

    if transformations is None:
        transformations = standard_transformations
    elif transformations == 'all':
        transformations = T[:]
    elif transformations == 'implicit':
        transformations = T[:6]
    elif isinstance(transformations, str):
        raise ValueError('unknown transformation group name')

    global_dict = dict(global_dict)
    global_dict['_sympy_parsing_div'] = Div
    code = stringify_expr(s, local_dict, global_dict, transformations)
    compiled = compile(evaluate_structural(code), '<string>', 'eval')
    try:
        return eval_expr(compiled, local_dict, global_dict)
    finally:
        for name in local_dict.pop(null, ()):
            local_dict[name] = null
