import typing as t
from dataclasses import dataclass

import torch
import typing_extensions as t_ext
from torch.utils.data import default_collate as torch_default_collate_fn

from kaggle_toolbox.device import Device


class Movable(t_ext.Protocol):

    def to(self, device: Device) -> t_ext.Self:
        ...

    def cpu(self) -> t_ext.Self:
        ...


_T = t.TypeVar('_T')
_X = t.TypeVar('_X', bound=Movable)


@dataclass
class DatasetItem(t.Generic[_X]):
    id: t.List[str]
    x: _X


@dataclass
class LabeledDatasetItem(DatasetItem[_X]):
    y: torch.Tensor


def default_collate_fn(item_iter: t.Iterable[_T]) -> _T:
    return torch_default_collate_fn(item_iter if isinstance(item_iter, list) else list(item_iter))


def default_id_collate_fn(id_list: t.Iterable[t.List[str]]) -> t.List[str]:
    return sum(id_list, [])


class DatasetItemCollator(t.Generic[_X]):

    def __init__(
            self,
            x_collate_fn: t.Callable[[t.Iterable[_X]], _X],
            id_collate_fn: t.Callable[[t.Iterable[t.List[str]]], t.List[str]] = default_id_collate_fn):
        self._x_collate_fn = x_collate_fn
        self._id_collate_fn = id_collate_fn

    def _collate_x(self, item_list: t.Iterable[DatasetItem[_X]]) -> _X:
        return self._x_collate_fn([item.x for item in item_list])

    def _collate_id(self, item_list: t.Iterable[DatasetItem[_X]]) -> t.List[str]:
        return self._id_collate_fn([item.id for item in item_list])

    def __call__(self, item_list: t.Iterable[DatasetItem[_X]]) -> DatasetItem[_X]:
        return DatasetItem(
            id=self._collate_id(item_list),
            x=self._collate_x(item_list))


class LabeledDatasetItemCollator(DatasetItemCollator[_X]):

    def __init__(
            self,
            x_collate_fn: t.Callable[[t.Iterable[_X]], _X],
            id_collate_fn: t.Callable[[t.Iterable[t.List[str]]], t.List[str]] = default_id_collate_fn,
            y_collate_fn: t.Callable[[t.Iterable[torch.Tensor]], torch.Tensor] = default_collate_fn):
        super().__init__(x_collate_fn=x_collate_fn, id_collate_fn=id_collate_fn)
        self._y_collate_fn = y_collate_fn

    def _collate_y(self, item_list: t.Iterable[LabeledDatasetItem[_X]]) -> torch.Tensor:
        return self._y_collate_fn([item.y for item in item_list])

    def __call__(self, item_list: t.Iterable[LabeledDatasetItem[_X]]) -> LabeledDatasetItem[_X]:
        return LabeledDatasetItem(
            id=self._collate_id(item_list),
            x=self._collate_x(item_list),
            y=self._collate_y(item_list))
