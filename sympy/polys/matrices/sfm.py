"""Sparse fill matrices over a domain."""

from __future__ import annotations

from operator import add, mul, neg, sub
from typing import TYPE_CHECKING, Generic

from sympy.polys.domains.domain import Er, Es

from .ddm import DDM
from .exceptions import DMBadInputError, DMDomainError, DMShapeError
from .sdm import SDM, sdm_transpose

if TYPE_CHECKING:
    from typing import Callable, Iterator

    from sympy.polys.domains.domain import Domain


class SFM(Generic[Er]):
    """A matrix with a common fill value and sparsely stored exceptions.

    Like SDM, the entries are stored in a dictionary of dictionaries. Unlike
    SDM, a missing entry represents the fill value rather than zero, and a
    stored entry can be zero. The entries are actual values, not differences
    from the fill.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices.sfm import SFM
    >>> A = SFM.ones((2, 3), ZZ)
    >>> A.setitem(0, 1, ZZ(0))
    >>> A.setitem(1, 2, ZZ(5))
    >>> A.to_list()
    [[1, 0, 1], [1, 1, 5]]
    """

    def __init__(
        self,
        elemsdict: dict[int, dict[int, Er]],
        shape: tuple[int, int],
        domain: Domain[Er],
        fill: Er,
    ) -> None:
        self.shape = self.rows, self.cols = m, n = shape
        self.domain = domain
        self.fill = fill
        self._data: dict[int, dict[int, Er]] = {}

        for i, row in elemsdict.items():
            if not 0 <= i < m:
                raise DMBadInputError("Row out of range")
            if not all(0 <= j < n for j in row):
                raise DMBadInputError("Column out of range")
            exceptions = {j: e for j, e in row.items() if e != fill}
            if exceptions:
                self._data[i] = exceptions

    @classmethod
    def const(cls, shape: tuple[int, int], domain: Domain[Es], fill: Es) -> SFM[Es]:
        """Return a constant matrix without storing its individual entries."""
        return cls({}, shape, domain, fill)

    @classmethod
    def zeros(cls, shape: tuple[int, int], domain: Domain[Es]) -> SFM[Es]:
        return cls.const(shape, domain, domain.zero)

    @classmethod
    def ones(cls, shape: tuple[int, int], domain: Domain[Es]) -> SFM[Es]:
        return cls.const(shape, domain, domain.one)

    def getitem(self, i: int, j: int) -> Er:
        m, n = self.shape
        if not (-m <= i < m and -n <= j < n):
            raise IndexError("index out of range")
        return self._data.get(i % m, {}).get(j % n, self.fill)

    def setitem(self, i: int, j: int, value: Er) -> None:
        m, n = self.shape
        if not (-m <= i < m and -n <= j < n):
            raise IndexError("index out of range")
        i, j = i % m, j % n
        if value != self.fill:
            self._data.setdefault(i, {})[j] = value
        elif i in self._data:
            row = self._data[i]
            row.pop(j, None)
            if not row:
                del self._data[i]

    def copy(self) -> SFM[Er]:
        return SFM(self._data, self.shape, self.domain, self.fill)

    def to_list(self) -> list[list[Er]]:
        m, n = self.shape
        rows = [[self.fill] * n for _ in range(m)]
        for i, row in self._data.items():
            for j, value in row.items():
                rows[i][j] = value
        return rows

    def to_ddm(self) -> DDM[Er]:
        return DDM(self.to_list(), self.shape, self.domain)

    def to_sdm(self) -> SDM[Er]:
        if self.domain.is_zero(self.fill):
            return SDM({i: row.copy() for i, row in self._data.items()},
                       self.shape, self.domain)
        data: dict[int, dict[int, Er]] = {}
        for (i, j), value in self.iter_items():
            data.setdefault(i, {})[j] = value
        return SDM(data, self.shape, self.domain)

    def iter_items(self) -> Iterator[tuple[tuple[int, int], Er]]:
        """Iterate over nonzero entries, including implicit nonzero values."""
        K = self.domain
        if not self.rows or not self.cols:
            return
        if K.is_zero(self.fill):
            for i, row in self._data.items():
                for j, value in row.items():
                    yield (i, j), value
        else:
            for i in range(self.rows):
                row = self._data.get(i, {})
                for j in range(self.cols):
                    value = row.get(j, self.fill)
                    if not K.is_zero(value):
                        yield (i, j), value

    def iter_values(self) -> Iterator[Er]:
        for _, value in self.iter_items():
            yield value

    def nnz(self) -> int:
        """Number of nonzero entries, without expanding the fill."""
        if self.domain.is_zero(self.fill):
            return sum(map(len, self._data.values()))
        nzeros = sum(
            self.domain.is_zero(e)
            for row in self._data.values() for e in row.values()
        )
        return self.rows * self.cols - nzeros

    def is_zero_matrix(self) -> bool:
        return self.nnz() == 0

    def transpose(self) -> SFM[Er]:
        data = sdm_transpose(self._data)
        return SFM(data, (self.cols, self.rows), self.domain, self.fill)

    def applyfunc(self, func: Callable[[Er], Es], domain: Domain[Es]) -> SFM[Es]:
        """Apply a function to the fill and each explicitly stored entry."""
        if sum(map(len, self._data.values())) == self.rows * self.cols:
            # The fill is not an entry of this matrix.
            fill = domain.zero
        else:
            fill = func(self.fill)
        data = {i: {j: func(e) for j, e in row.items()}
                for i, row in self._data.items()}
        return SFM(data, self.shape, domain, fill)

    def convert_to(self, domain: Domain[Es]) -> SFM[Es]:
        return self.applyfunc(lambda e: domain.convert_from(e, self.domain), domain)

    def neg(self) -> SFM[Er]:
        return self.applyfunc(neg, self.domain)

    def mul(self, b: Er) -> SFM[Er]:
        return self.applyfunc(lambda e: e * b, self.domain)

    def rmul(self, b: Er) -> SFM[Er]:
        return self.applyfunc(lambda e: b * e, self.domain)

    def _binop(
        self, B: SFM[Er] | SDM[Er], op: Callable[[Er, Er], Er]
    ) -> SFM[Er]:
        if self.domain != B.domain:
            raise DMDomainError
        if self.shape != B.shape:
            raise DMShapeError

        A_data, A_fill = self._data, self.fill
        if isinstance(B, SDM):
            B_data, B_fill = B, B.domain.zero
        else:
            B_data, B_fill = B._data, B.fill

        fill = op(A_fill, B_fill)
        data = {}
        for i in A_data.keys() | B_data.keys():
            Ai, Bi = A_data.get(i, {}), B_data.get(i, {})
            row = {}
            for j in Ai.keys() | Bi.keys():
                value = op(Ai.get(j, A_fill), Bi.get(j, B_fill))
                if value != fill:
                    row[j] = value
            if row:
                data[i] = row
        return SFM(data, self.shape, self.domain, fill)

    def add(self, B: SFM[Er] | SDM[Er]) -> SFM[Er]:
        return self._binop(B, add)

    def sub(self, B: SFM[Er] | SDM[Er]) -> SFM[Er]:
        return self._binop(B, sub)

    def mul_elementwise(self, B: SFM[Er] | SDM[Er]) -> SFM[Er]:
        return self._binop(B, mul)
