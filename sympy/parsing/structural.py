"""Experimental structural arithmetic used by the optional parser mode.

Mul2 and Div are binary expressions which never evaluate on construction.
They preserve the distinction between writing a*b**-1 and a/b, as well as
the order of successive multiplications. Call touch() to lower them into
ordinary SymPy arithmetic.
"""

from __future__ import annotations
import ast

from sympy.core.function import Function
from sympy.core.sympify import sympify
from sympy.parsing.sympy_parser import EvaluateFalseTransformer


def _operand_str(printer, arg):
    result = printer.doprint(arg)
    if (arg.is_Add or isinstance(arg, (Mul2, Div)) or
            (arg.is_Number and arg.is_negative)):
        return '(%s)' % result
    return result


def _operand_latex(printer, arg):
    result = printer.doprint(arg)
    if arg.is_Add or isinstance(arg, (Mul2, Div)):
        return r'\left(%s\right)' % result
    return result


class Mul2(Function):
    """An unevaluated binary multiplication operation."""
    nargs = 2

    def doit(self, deep=True, **hints):
        a, b = self.args
        if deep:
            a, b = (arg.doit(deep=True, **hints) for arg in (a, b))
        return a*b

    def _sympystr(self, printer):
        a, b = self.args
        return '%s*%s' % (_operand_str(printer, a),
                          _operand_str(printer, b))

    def _latex(self, printer):
        a, b = self.args
        return r'%s \cdot %s' % (_operand_latex(printer, a),
                                _operand_latex(printer, b))


class Div(Function):
    """An unevaluated binary division operation."""
    nargs = 2

    def doit(self, deep=True, **hints):
        a, b = self.args
        if deep:
            a, b = (arg.doit(deep=True, **hints) for arg in (a, b))
        return a/b

    def _sympystr(self, printer):
        a, b = self.args
        return '%s/%s' % (_operand_str(printer, a),
                          _operand_str(printer, b))

    def _latex(self, printer):
        a, b = self.args
        return r'\frac{%s}{%s}' % (printer.doprint(a),
                                  printer.doprint(b))


def touch(expr):
    """Evaluate structural multiplication and division, inside-out.

    Unlike a general doit(), this only asks Mul2 and Div to perform their
    operations; it does not explicitly evaluate unrelated functions.
    """
    return sympify(expr).replace(
        lambda e: isinstance(e, (Mul2, Div)),
        lambda e: e.doit(deep=False))


class StructuralTransformer(EvaluateFalseTransformer):
    """Retain binary multiplication and division in parse_expr's AST."""

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.Mult, ast.Div)):
            name = ('_sympy_parsing_mul2' if isinstance(node.op, ast.Mult)
                    else '_sympy_parsing_div')
            return ast.Call(
                func=ast.Name(id=name, ctx=ast.Load()),
                args=[self.visit(node.left), self.visit(node.right)],
                keywords=[])
        return super().visit_BinOp(node)


def evaluate_structural(s):
    """Return an AST retaining multiply/divide operations without flattening."""
    node = ast.parse(s)
    transformed = StructuralTransformer().visit(node)
    transformed = ast.Expression(transformed.body[0].value)
    return ast.fix_missing_locations(transformed)


def parse_structural_expr(s, local_dict=None, transformations=None,
                          global_dict=None):
    """Experimentally parse multiplication and division as binary operations.

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
    global_dict['_sympy_parsing_mul2'] = Mul2
    global_dict['_sympy_parsing_div'] = Div
    code = stringify_expr(s, local_dict, global_dict, transformations)
    compiled = compile(evaluate_structural(code), '<string>', 'eval')
    try:
        return eval_expr(compiled, local_dict, global_dict)
    finally:
        for name in local_dict.pop(null, ()):
            local_dict[name] = null
