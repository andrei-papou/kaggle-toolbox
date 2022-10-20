from __future__ import annotations

import itertools
import typing as t

import typing_extensions as t_ext

from kaggle_toolbox.iter.torch_types import DataLoader

_T = t.TypeVar('_T', covariant=True)
_L = t.TypeVar('_L', covariant=True)


class SizedIter(t.Generic[_T]):

    def __init__(self, it: t.Iterator[_T], n: int) -> None:
        self._it = it
        self._n = n

    @classmethod
    def from_data_loader(cls, data_loader: DataLoader[_T]) -> t_ext.Self:
        return cls(it=iter(data_loader), n=len(data_loader))

    def __iter__(self) -> t_ext.Self:
        return self

    def __len__(self) -> int:
        return self._n

    def __next__(self) -> _T:
        if self._n <= 0:
            raise StopIteration()
        self._n -= 1
        return next(self._it)

    @property
    def is_empty(self) -> bool:
        return self._n <= 0

    def zip(self, it: SizedIter[_L]) -> SizedIter[t.Tuple[_T, _L]]:
        assert self._n == len(it)
        return SizedIter(
            it=zip(iter(self), iter(it)),
            n=self._n)

    def chain(self, it: SizedIter[_T]) -> SizedIter[_T]:
        return SizedIter(
            it=itertools.chain(iter(self), iter(it)),
            n=self._n + len(it))
