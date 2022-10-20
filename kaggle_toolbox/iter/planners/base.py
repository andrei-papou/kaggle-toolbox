
import typing as t

from kaggle_toolbox.iter.index import Index
from kaggle_toolbox.iter.sized_iter import SizedIter
from kaggle_toolbox.iter.torch_types import DataLoader

_T = t.TypeVar('_T', covariant=True)


class IterPlanner(t.Generic[_T]):

    def __init__(self, data_loader: DataLoader[_T]) -> None:
        self._data_loader = data_loader

    @property
    def epoch(self) -> int:
        raise NotImplementedError()

    @property
    def step(self) -> int:
        raise NotImplementedError()

    def get_next_iter(self, val_metric: t.Optional[float] = None) -> SizedIter[t.Tuple[Index, _T]]:
        raise NotImplementedError()


class IterPlannerBuilder:

    def build(self, data_loader: DataLoader[_T]) -> IterPlanner[_T]:
        raise NotImplementedError()
