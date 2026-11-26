from __future__ import annotations

from sympy import QQ, ZZ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.sfm import SFM
from sympy.polys.matrices.sdm import SDM
from sympy.testing.pytest import raises


def test_SFM_const():
    shape = (10**6, 10**6)
    for A, value in [
        (SFM.zeros(shape, ZZ), ZZ.zero),
        (SFM.ones(shape, ZZ), ZZ.one),
        (SFM.const(shape, ZZ, ZZ(7)), ZZ(7)),
    ]:
        assert A.shape == shape
        assert A.domain == ZZ
        assert A.fill == value
        assert A._data == {}
        assert A.getitem(0, 0) == A.getitem(-1, -1) == value


def test_SFM_get_set_copy():
    A = SFM.ones((2, 3), ZZ)
    A.setitem(0, 1, ZZ.zero)
    A.setitem(-1, -1, ZZ(5))
    assert A.getitem(0, 1) == ZZ.zero
    assert A.getitem(1, 2) == ZZ(5)
    assert A._data == {0: {1: ZZ.zero}, 1: {2: ZZ(5)}}

    B = A.copy()
    A.setitem(0, 1, ZZ.one)
    A.setitem(1, 2, ZZ.one)
    assert A._data == {}
    assert B.getitem(0, 1) == ZZ.zero
    assert B.getitem(1, 2) == ZZ(5)

    for i, j in [(2, 0), (0, 3), (-3, 0), (0, -4)]:
        raises(IndexError, lambda: A.getitem(i, j))
        raises(IndexError, lambda: A.setitem(i, j, ZZ.zero))


def test_SFM_constructor():
    data = {0: {0: QQ.one, 1: QQ.zero}, 1: {2: QQ(2, 3)}}
    A = SFM(data, (2, 3), QQ, QQ.one)
    assert A._data == {0: {1: QQ.zero}, 1: {2: QQ(2, 3)}}
    data[0][1] = QQ(7)
    assert A.getitem(0, 1) == QQ.zero
    raises(DMBadInputError, lambda: SFM({2: {}}, (2, 3), QQ, QQ.one))
    raises(DMBadInputError, lambda: SFM({0: {3: QQ.one}}, (2, 3), QQ, QQ.one))


def test_SFM_conversion():
    A = SFM({0: {1: ZZ.zero}, 1: {2: ZZ(5)}}, (2, 3), ZZ, ZZ.one)
    expected = [[ZZ.one, ZZ.zero, ZZ.one], [ZZ.one, ZZ.one, ZZ(5)]]
    assert A.to_list() == expected
    assert A.to_ddm() == DDM(expected, (2, 3), ZZ)
    assert A.to_sdm() == DDM(expected, (2, 3), ZZ).to_sdm()

    A = SFM({0: {1: ZZ(2)}}, (2, 3), ZZ, ZZ.zero)
    B = A.to_sdm()
    assert B == SDM({0: {1: ZZ(2)}}, (2, 3), ZZ)
    B.setitem(0, 1, ZZ.zero)
    assert A.getitem(0, 1) == ZZ(2)

    for shape in [(0, 0), (0, 3), (2, 0)]:
        A = SFM.ones(shape, ZZ)
        assert A.to_ddm().shape == A.to_sdm().shape == shape
        raises(IndexError, lambda: A.getitem(0, 0))
