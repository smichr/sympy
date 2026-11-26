from __future__ import annotations

from sympy import EXRAW, GF, QQ, ZZ, nan, oo
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import DMBadInputError, DMDomainError, DMShapeError
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


def test_SFM_nonzero_entries():
    A = SFM({0: {1: ZZ.zero}, 1: {2: ZZ(5)}}, (2, 3), ZZ, ZZ.one)
    assert A.nnz() == 5
    assert dict(A.iter_items()) == {
        (0, 0): ZZ.one, (0, 2): ZZ.one,
        (1, 0): ZZ.one, (1, 1): ZZ.one, (1, 2): ZZ(5),
    }
    assert list(A.iter_values()) == [ZZ.one]*4 + [ZZ(5)]
    assert not A.is_zero_matrix()
    assert A.transpose().to_ddm() == A.to_ddm().transpose()
    assert A.transpose()._data == {1: {0: ZZ.zero}, 2: {1: ZZ(5)}}

    B = SFM({0: {1: ZZ(2)}}, (2, 3), ZZ, ZZ.zero)
    assert B.nnz() == 1
    assert list(B.iter_items()) == [((0, 1), ZZ(2))]
    assert list(B.iter_values()) == [ZZ(2)]
    assert SFM.zeros((2, 3), ZZ).is_zero_matrix()
    assert SFM.ones((2, 0), ZZ).is_zero_matrix()
    assert list(SFM.ones((10**12, 0), ZZ).iter_items()) == []
    assert SFM({0: {0: ZZ.zero}}, (1, 1), ZZ, ZZ.one).is_zero_matrix()


def test_SFM_applyfunc():
    A = SFM({0: {1: ZZ(2)}}, (10**6, 10**6), ZZ, ZZ(3))
    B = A.convert_to(QQ).applyfunc(lambda e: QQ.one/e, QQ)
    assert B.domain == QQ
    assert B.fill == QQ(1, 3)
    assert B._data == {0: {1: QQ(1, 2)}}
    assert B.nnz() == 10**12

    A = SFM({0: {1: QQ(2), 2: QQ.zero}}, (1, 3), QQ, QQ.one)
    B = A.applyfunc(lambda e: e**2-e, QQ)
    assert B.fill == QQ.zero
    assert B._data == {0: {1: QQ(2)}}

    C = SFM({0: {1: ZZ(3)}}, (1, 2), ZZ, ZZ.one).convert_to(GF(2))
    assert C.fill == C.domain.one
    assert C._data == {}

    # An unused fill must not be passed to the function.
    A = SFM({0: {0: QQ(2)}}, (1, 1), QQ, QQ.zero)
    assert A.applyfunc(lambda e: QQ.one/e, QQ).to_list() == [[QQ(1, 2)]]
    A = SFM.zeros((0, 2), QQ)
    assert A.applyfunc(lambda e: QQ.one/e, QQ).shape == (0, 2)


def test_SFM_arithmetic():
    A = SFM({0: {1: ZZ.zero}, 1: {0: ZZ(3)}}, (2, 2), ZZ, ZZ.one)
    B = SFM({0: {0: ZZ(5)}, 1: {0: ZZ.zero}}, (2, 2), ZZ, ZZ(2))
    Ad, Bd = A.to_ddm(), B.to_ddm()
    assert A.add(B).to_ddm() == Ad.add(Bd)
    assert A.sub(B).to_ddm() == Ad.sub(Bd)
    assert A.mul_elementwise(B).to_ddm() == Ad.mul_elementwise(Bd)
    assert A.neg().to_ddm() == Ad.neg()
    assert A.mul(ZZ(3)).to_ddm() == Ad.mul(ZZ(3))
    assert A.rmul(ZZ(3)).to_ddm() == Ad.rmul(ZZ(3))
    assert A.mul(ZZ.zero).is_zero_matrix()

    B = SDM({0: {1: ZZ(5)}, 1: {0: ZZ(4)}}, (2, 2), ZZ)
    assert A.add(B).to_ddm() == Ad.add(B.to_ddm())
    assert A.sub(B).to_ddm() == Ad.sub(B.to_ddm())
    assert A.mul_elementwise(B).to_ddm() == Ad.mul_elementwise(B.to_ddm())

    raises(DMDomainError, lambda: A.add(SFM.ones((2, 2), QQ)))
    raises(DMShapeError, lambda: A.add(SFM.ones((2, 3), ZZ)))

    A = SFM.const((1, 2), EXRAW, oo)
    B = SDM({0: {0: EXRAW.one}}, (1, 2), EXRAW)
    C = A.mul_elementwise(B)
    assert C.fill == nan
    assert C.to_list() == [[oo, nan]]


def test_SFM_sparse_denominators():
    shape = (10**6, 10**6)
    numers = SDM({0: {3: QQ.one}, 1: {5: QQ(2)}}, shape, QQ)
    denoms = SFM({0: {3: QQ(3)}}, shape, QQ, QQ.one)
    inv = denoms.applyfunc(lambda e: QQ.one/e, QQ)
    assert inv.fill == QQ.one
    assert inv._data == {0: {3: QQ(1, 3)}}
    result = inv.mul_elementwise(numers)
    assert result.fill == QQ.zero
    assert result.to_sdm() == SDM({0: {3: QQ(1, 3)}, 1: {5: QQ(2)}}, shape, QQ)
