"""Experimental structural arithmetic used by the optional parser mode.

Ordinary unevaluated Add, Mul, and Pow preserve operation order and nesting.
A held negative exponent distinguishes a written inverse power from the
reciprocal Pow used to represent division. Call touch() to evaluate arithmetic.
"""

from __future__ import annotations
import ast

from sympy.core.add import Add
from sympy.core.expr import UnevaluatedExpr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sympify import sympify
from sympy.parsing.sympy_parser import EvaluateFalseTransformer


def touch(expr):
    """Evaluate arithmetic in nesting order without invoking unrelated doit's.

    Release an explicitly held exponent as part of evaluating its Pow.
    """
    def evaluate(e):
        if isinstance(e, Pow) and isinstance(e.exp, UnevaluatedExpr):
            return Pow(e.base, e.exp.args[0])
        return e.func(*e.args)

    return sympify(expr).replace(
        lambda e: isinstance(e, (Add, Mul, Pow)), evaluate)


class StructuralTransformer(EvaluateFalseTransformer):
    """Preserve binary arithmetic grouping and distinguish written powers."""

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, ast.Sub):
                # Preserve a - (b + c) as a binary Add with a negated group.
                right = ast.Call(
                    func=ast.Name(id='Mul', ctx=ast.Load()),
                    args=[ast.UnaryOp(
                        op=ast.USub(), operand=ast.Constant(1)), right],
                    keywords=[ast.keyword(
                        arg='evaluate', value=ast.Constant(False))])
            elif isinstance(node.op, ast.Div):
                # Represent a written division using a reciprocal Pow.
                right = ast.Call(
                    func=ast.Name(id='Pow', ctx=ast.Load()),
                    args=[right, ast.UnaryOp(
                        op=ast.USub(), operand=ast.Constant(1))],
                    keywords=[ast.keyword(
                        arg='evaluate', value=ast.Constant(False))])
            elif (isinstance(node.op, ast.Pow) and
                  isinstance(node.right, ast.UnaryOp) and
                  isinstance(node.right.op, ast.USub)):
                # A written negative power must not print as division.
                right = ast.Call(
                    func=ast.Name(id='_sympy_parsing_hold_exp', ctx=ast.Load()),
                    args=[right], keywords=[])
            name = ('Pow' if isinstance(node.op, ast.Pow) else
                    'Add' if isinstance(node.op, (ast.Add, ast.Sub)) else 'Mul')
            return ast.Call(
                func=ast.Name(id=name, ctx=ast.Load()),
                args=[left, right],
                keywords=[ast.keyword(
                    arg='evaluate', value=ast.Constant(False))])
        return super().visit_BinOp(node)


def evaluate_structural(s):
    """Return an AST preserving arithmetic nesting without flattening."""
    node = ast.parse(s)
    transformed = StructuralTransformer().visit(node)
    transformed = ast.Expression(transformed.body[0].value)
    return ast.fix_missing_locations(transformed)


def parse_structural_expr(s, local_dict=None, transformations=None,
                          global_dict=None):
    """Experimentally parse arithmetic expressions without flattening.

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
    global_dict['_sympy_parsing_hold_exp'] = UnevaluatedExpr
    code = stringify_expr(s, local_dict, global_dict, transformations)
    compiled = compile(evaluate_structural(code), '<string>', 'eval')
    try:
        return eval_expr(compiled, local_dict, global_dict)
    finally:
        for name in local_dict.pop(null, ()):
            local_dict[name] = null
