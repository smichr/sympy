"""Experimental, operator-annotated structural parsing.

The intermediate representation is intended for printing and touch(), not for
ordinary algebra: annotations are genuine Symbols until touch() removes them.
As with parse_expr, do not parse untrusted input with this module.
"""
from __future__ import annotations

import ast
import builtins
import types

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.parsing.sympy_parser import (
    EvaluateFalseTransformer, T, eval_expr, null, standard_transformations,
    stringify_expr,
)

_TAGS = {
    name: Symbol('', **{'parse_' + name: True})
    for name in ('addition', 'subtraction', 'multiplication',
                 'division', 'power', 'negation')
}


def annotation(expr):
    """Identify a tagged two-argument Mul; return (operator, payload) or None.

    The tag has exactly two assumptions, 'commutative': True and
    'parse_<operator>': True. Do not recognize generic empty Symbols.
    """
    if not (getattr(expr, 'is_Mul', False) and len(expr.args) == 2):
        return None
    tag, payload = expr.args
    if not isinstance(tag, Symbol) or tag.name:
        return None
    assumptions = tag.assumptions0
    if len(assumptions) != 2 or assumptions.get('commutative') is not True:
        return None
    keys = [k for k, v in assumptions.items()
            if k.startswith('parse_') and v is True]
    if len(keys) != 1:
        return None
    return keys[0][len('parse_'):], payload


def tagged(operator, payload):
    """Annotate only where ordinary unevaluated printing loses structure.

    Simple additions and multiplications already preserve their grouping.
    Keep the marker when their textual forms can disguise a written operator,
    including a positive Add with a negative term, a negative numeric factor,
    and multiplication by an explicitly written power.
    """
    if operator == 'addition':
        left, right = payload.args
        if ((right.is_Symbol and right.name) or
                (right.is_Integer and right.is_nonnegative)):
            return payload
    elif operator == 'multiplication':
        left, right = payload.args
        signed_numeric = any(
            arg.is_Number and arg.is_negative for arg in payload.args)
        right_annotation = annotation(right)
        if not signed_numeric and not (
                right_annotation and right_annotation[0] == 'power'):
            return payload
    return Mul(_TAGS[operator], payload, evaluate=False)


def touch(expr):
    """Remove annotations *before* evaluating children, without calling doit()."""
    expr = sympify(expr)
    ann = annotation(expr)
    if ann is not None:
        return touch(ann[1])
    if not expr.args:
        return expr
    args = tuple(touch(a) if hasattr(a, 'args') else a for a in expr.args)
    if isinstance(expr, (Add, Mul, Pow)):
        return expr.func(*args)
    if args == expr.args:
        return expr
    return expr.func(*args)


def _call(func, *args):
    return ast.Call(func=ast.Name(id=func, ctx=ast.Load()),
                    args=list(args), keywords=[])


def _arithmetic(func, *args):
    return ast.Call(
        func=ast.Name(id=func, ctx=ast.Load()), args=list(args),
        keywords=[ast.keyword(arg='evaluate', value=ast.Constant(False))])


def _tag(operator, payload):
    return _call('_sympy_parsing_tag', ast.Constant(operator), payload)


class TaggedStructuralTransformer(EvaluateFalseTransformer):
    """Preserve each written arithmetic operator as a tagged binary tree."""

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub):
            operand = self.visit(node.operand)
            # A signed numeric literal remains a number. In particular,
            # -1**2 is *not* a literal: the operand is the entire power.
            if (isinstance(node.operand, ast.Call)
                    and isinstance(node.operand.func, ast.Name)
                    and node.operand.func.id in ('Integer', 'Float')):
                return ast.UnaryOp(op=ast.USub(), operand=operand)
            if (isinstance(node.operand, ast.Constant)
                    and type(node.operand.value) in (int, float)):
                return ast.UnaryOp(op=ast.USub(), operand=operand)
            return _tag('negation', _arithmetic(
                'Mul', ast.Constant(-1), operand))
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if not isinstance(node.op, (
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            return super().visit_BinOp(node)

        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return _tag('addition', _arithmetic('Add', left, right))
        if isinstance(node.op, ast.Sub):
            return _tag('subtraction', _arithmetic(
                'Add', left, _arithmetic('Mul', ast.Constant(-1), right)))
        if isinstance(node.op, ast.Mult):
            return _tag('multiplication', _arithmetic('Mul', left, right))
        if isinstance(node.op, ast.Div):
            return _tag('division', _arithmetic(
                'Mul', left, _arithmetic('Pow', right, ast.Constant(-1))))
        return _tag('power', _arithmetic('Pow', left, right))


def tagged_ast(code):
    """Convert token-transformed Python source into structural arithmetic AST."""
    transformed = TaggedStructuralTransformer().visit(ast.parse(code))
    node = ast.Expression(transformed.body[0].value)
    return ast.fix_missing_locations(node)


def parse_tagged_expr(s, local_dict=None, transformations=None, global_dict=None):
    """Parse an expression with structural operator tags (experimental).

    Parsing requires eval, and must not be used on untrusted input.
    """
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
    global_dict['_sympy_parsing_tag'] = tagged
    code = stringify_expr(s, local_dict, global_dict, transformations)
    compiled = compile(tagged_ast(code), '<string>', 'eval')
    try:
        return eval_expr(compiled, local_dict, global_dict)
    finally:
        for name in local_dict.pop(null, ()):
            local_dict[name] = null
