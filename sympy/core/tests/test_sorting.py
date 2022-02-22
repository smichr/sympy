from sympy.core.sorting import default_sort_key, ordered, repsort
from sympy.testing.pytest import raises

from sympy.abc import x, y, z


def test_default_sort_key():
    func = lambda x: x
    assert sorted([func, x, func], key=default_sort_key) == [func, func, x]

    class C:
        def __repr__(self):
            return 'x.y'
    func = C()
    assert sorted([x, func], key=default_sort_key) == [func, x]


def test_ordered():
    # Issue 7210 - this had been failing with python2/3 problems
    assert (list(ordered([{1:3, 2:4, 9:10}, {1:3}])) == \
               [{1: 3}, {1: 3, 2: 4, 9: 10}])
    # warnings should not be raised for identical items
    l = [1, 1]
    assert list(ordered(l, warn=True)) == l
    l = [[1], [2], [1]]
    assert list(ordered(l, warn=True)) == [[1], [1], [2]]
    raises(ValueError, lambda: list(ordered(['a', 'ab'], keys=[lambda x: x[0]],
        default=False, warn=True)))


def test_repsort():
    assert repsort((x, y + 1), (z, x + 2)) == [(z, x + 2), (x, y + 1)]
    assert repsort((x, y + 1), (z, x**2)) == [(z, x**2), (x, y + 1)]
    raises(ValueError, lambda: repsort((x, y), (y, z), (z, x)))
