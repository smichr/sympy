"""Sparse fill matrices over a domain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from sympy.polys.domains.domain import Er, Es

from .ddm import DDM
from .exceptions import DMBadInputError
from .sdm import SDM

if TYPE_CHECKING:
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
        return self.to_ddm().to_sdm()
