from sympy import S, symbols

from sympy.polys.recurrences import DifferentialRecurrence


def test_differential_recurrence_affine_product():
    n = symbols('n', integer=True)

    # F(t) = (2 + 3*t)**40 * (5 - 7*t)**30 satisfies
    #
    #   (10 + t - 21*t**2)*F' - (180 - 1470*t)*F = 0.
    recurrence = DifferentialRecurrence((
        (0, 0, -180),
        (0, 1, 1470),
        (1, 0, 10),
        (1, 1, 1),
        (1, 2, -21),
    ))

    assert recurrence.shifts == (-1, 0, 1)
    assert recurrence.coefficient(-1, n).expand() == 1491 - 21*n
    assert recurrence.coefficient(0, n).expand() == n - 180
    assert recurrence.coefficient(1, n).expand() == 10*n + 10


def test_differential_recurrence_terms():
    n = symbols('n', integer=True)
    recurrence = DifferentialRecurrence((
        (0, 0, S(2)),
        (1, 0, S(3)),
        (1, 1, S(5)),
    ))

    terms = recurrence.terms(n)
    assert terms[(0, 0)] == 2
    assert terms[(1, 0)] == 3*(n + 1)
    assert terms[(0, 1)] == 5*n
