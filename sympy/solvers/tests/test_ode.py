from sympy import (acos, acosh, asinh, atan, cos, Derivative, diff, dsolve,
    Dummy, Eq, Ne, erf, erfi, exp, Function, I, Integral, LambertW, log, O, pi,
    Rational, rootof, S, simplify, sin, sqrt, Subs, Symbol, tan, asin, sinh,
    Piecewise, symbols, Poly, sec, Ei, re, im)
from sympy.solvers.ode import (_undetermined_coefficients_match,
    checkodesol, classify_ode, classify_sysode, constant_renumber,
    constantsimp, homogeneous_order, infinitesimals, checkinfsol,
    checksysodesol, solve_ics, dsolve, get_numbered_constants)
from sympy.solvers.deutils import ode_order
from sympy.utilities.pytest import XFAIL, skip, raises, slow, ON_TRAVIS
from sympy.utilities.misc import filldedent


C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10 = symbols('C0:11')
u, x, y, z = symbols('u,x:z', real=True)
f = Function('f')
g = Function('g')
h = Function('h')



#@slow
def test_1st_homogeneous_coeff_ode():
    # Type: First order homogeneous, y'=f(y/x)
    eq1 = f(x)/x*cos(f(x)/x) - (x/f(x)*sin(f(x)/x) + cos(f(x)/x))*f(x).diff(x)
    eq2 = x*f(x).diff(x) - f(x) - x*sin(f(x)/x)
    eq3 = f(x) + (x*log(f(x)/x) - 2*x)*diff(f(x), x)
    eq4 = 2*f(x)*exp(x/f(x)) + f(x)*f(x).diff(x) - 2*x*exp(x/f(x))*f(x).diff(x)
    eq5 = 2*x**2*f(x) + f(x)**3 + (x*f(x)**2 - 2*x**3)*f(x).diff(x)
    eq6 = x*exp(f(x)/x) - f(x)*sin(f(x)/x) + x*sin(f(x)/x)*f(x).diff(x)
    eq7 = (x + sqrt(f(x)**2 - x*f(x)))*f(x).diff(x) - f(x)
    eq8 = x + f(x) - (x - f(x))*f(x).diff(x)

    sol1 = Eq(log(x), C1 - log(f(x)*sin(f(x)/x)/x))
    sol2 = Eq(log(x), log(C1) + log(cos(f(x)/x) - 1)/2 - log(cos(f(x)/x) + 1)/2)
    sol3 = Eq(f(x), -exp(C1)*LambertW(-x*exp(-C1 + 1)))
    sol4 = Eq(log(f(x)), C1 - 2*exp(x/f(x)))
    sol5 = Eq(f(x), exp(2*C1 + LambertW(-2*x**4*exp(-4*C1))/2)/x)
    sol6 = Eq(log(x), C1 + exp(-f(x)/x)*sin(f(x)/x)/2 + exp(-f(x)/x)*cos(f(x)/x)/2)
    sol7 = Eq(log(f(x)), C1 - 2*sqrt(-x/f(x) + 1))
    sol8 = Eq(log(x), C1 - log(sqrt(1 + f(x)**2/x**2)) + atan(f(x)/x))

    # indep_div_dep actually has a simpler solution for eq2,
    # but it runs too slow
    print( dsolve(eq5, hint='1st_homogeneous_coeff_best'), sol5)
