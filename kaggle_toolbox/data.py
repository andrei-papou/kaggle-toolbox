import typing as t
from dataclasses import dataclass

import torch
import typing_extensions as t_ext
from torch.utils.data import default_collate as default_collate_fn

from kaggle_toolbox.device import Device


class Movable(t_ext.Protocol):

    def to(self, device: Device) -> t_ext.Self:
        ...

    def cpu(self) -> t_ext.Self:
        ...


_X = t.TypeVar('_X', bound=Movable)


@dataclass
class DatasetItem(t.Generic[_X]):
    id: t.List[str]
    x: _X
    y: torch.Tensor



class DatasetItemCollator(t.Generic[_X]):

    def __init__(
            self,
            x_collate_fn: t.Callable[[t.List[_X]], _X],
            id_collate_fn: t.Callable[[t.List[t.List[str]]], t.List[str]] = default_collate_fn,
            y_collate_fn: t.Callable[[t.List[torch.Tensor]], torch.Tensor] = default_collate_fn):
        self._x_collate_fn = x_collate_fn
        self._id_collate_fn = id_collate_fn
        self._y_collate_fn = y_collate_fn

    def __call__(self, item_list: t.List[DatasetItem[_X]]) -> DatasetItem[_X]:
        return DatasetItem(
            id=self._id_collate_fn([item.id for item in item_list]),
            x=self._x_collate_fn([item.x for item in item_list]),
            y=self._y_collate_fn([item.y for item in item_list]))
