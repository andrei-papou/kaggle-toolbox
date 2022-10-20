import typing as t

from kaggle_toolbox.iter.index import Index, local_index_range
from kaggle_toolbox.iter.planners.base import IterPlanner, IterPlannerBuilder
from kaggle_toolbox.iter.sized_iter import SizedIter
from kaggle_toolbox.iter.subset_size import SubsetSize
from kaggle_toolbox.iter.torch_types import DataLoader
from kaggle_toolbox.metrics import MetricCriteria, sort_from_best_to_worst

_T = t.TypeVar('_T', covariant=True)


class _SubsetIterPlanner(IterPlanner[_T]):

    def __init__(
            self,
            data_loader: DataLoader[_T],
            stop_at_epoch_end: bool = True) -> None:
        super().__init__(data_loader)
        self._it = SizedIter.from_data_loader(data_loader)
        self._stop_at_epoch_end = stop_at_epoch_end
        self._epoch = 0
        self._step = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    def _rebuild_it(self):
        self._it = SizedIter.from_data_loader(self._data_loader)
        self._epoch += 1

    def _get_next_iter(self, subset_size: int) -> SizedIter[t.Tuple[Index, _T]]:
        self._step += 1
        if len(self._it) - subset_size < subset_size and self._stop_at_epoch_end:
            old_it = self._it
            self._rebuild_it()
            return local_index_range(
                self._epoch - 1,
                len(self._data_loader) - len(old_it),
                len(self._data_loader),
                len(self._data_loader)).zip(old_it)
        if len(self._it) < subset_size:
            old_it = self._it
            self._rebuild_it()
            return local_index_range(
                self._epoch - 1,
                len(self._data_loader) - len(old_it),
                len(self._data_loader),
                len(self._data_loader)
            ).zip(old_it).chain(
                local_index_range(
                    self._epoch,
                    0,
                    subset_size - len(old_it),
                    len(self._data_loader)
                ).zip(SizedIter(self._it, subset_size - len(old_it))))
        return local_index_range(
            self._epoch,
            len(self._data_loader) - len(self._it),
            len(self._data_loader) - len(self._it) + subset_size,
            len(self._data_loader)).zip(SizedIter(self._it, subset_size))


class FixedSubsetIterPlanner(_SubsetIterPlanner[_T]):

    def __init__(
            self,
            data_loader: DataLoader[_T],
            subset_size: int,
            stop_at_epoch_end: bool = True) -> None:
        super().__init__(data_loader, stop_at_epoch_end)
        self._subset_size = subset_size

    def get_next_iter(self, val_metric: t.Optional[float] = None) -> SizedIter[t.Tuple[Index, _T]]:
        return self._get_next_iter(subset_size=self._subset_size)


class FixedSubsetIterPlannerBuilder(IterPlannerBuilder):

    def __init__(
            self,
            subset_size: SubsetSize,
            stop_at_epoch_end: bool = True):
        self._subset_size = subset_size
        self._stop_at_epoch_end = stop_at_epoch_end

    def build(self, data_loader: DataLoader[_T]) -> IterPlanner[_T]:
        return FixedSubsetIterPlanner(
            data_loader=data_loader,
            subset_size=self._subset_size.as_nat(len(data_loader)),
            stop_at_epoch_end=self._stop_at_epoch_end)


class EpochBasedSubsetIterPlanner(_SubsetIterPlanner[_T]):

    def __init__(
            self,
            data_loader: DataLoader[_T],
            epoch_to_subset_size_mapping: t.Mapping[int, int],
            stop_at_epoch_end: bool = True) -> None:
        super().__init__(data_loader, stop_at_epoch_end)
        assert 0 in epoch_to_subset_size_mapping, 'Subset size  for epoch 0 must be provided explicitly.'
        self._epoch_to_subset_size_mapping = epoch_to_subset_size_mapping

    def _get_subset_size_for_epoch(self, epoch: int) -> int:
        for ep in reversed(range(0, epoch + 1)):
            if ep in self._epoch_to_subset_size_mapping:
                return self._epoch_to_subset_size_mapping[ep]
        raise RuntimeError('Subset size not found but it is guaranteed to be available for 0.')

    def get_next_iter(self, val_metric: t.Optional[float] = None) -> SizedIter[t.Tuple[Index, _T]]:
        return self._get_next_iter(self._get_subset_size_for_epoch(self._epoch))


class EpochBasedSubsetIterPlannerBuilder(IterPlannerBuilder):

    def __init__(
            self,
            epoch_to_subset_size_mapping: t.Mapping[int, SubsetSize],
            stop_at_epoch_end: bool = True):
        self._epoch_to_subset_size_mapping = epoch_to_subset_size_mapping
        self._stop_at_epoch_end = stop_at_epoch_end

    def build(self, data_loader: DataLoader[_T]) -> IterPlanner[_T]:
        return EpochBasedSubsetIterPlanner(
            data_loader=data_loader,
            epoch_to_subset_size_mapping={
                ep: ss.as_nat(len(data_loader))
                for ep, ss in self._epoch_to_subset_size_mapping.items()
            },
            stop_at_epoch_end=self._stop_at_epoch_end)


class MetricBasedSubsetIterPlanner(_SubsetIterPlanner[_T]):

    def __init__(
            self,
            data_loader: DataLoader[_T],
            metric_threshold_to_subset_size_mapping: t.Mapping[float, int],
            metric_criteria: MetricCriteria,
            stop_at_epoch_end: bool = True) -> None:
        super().__init__(data_loader, stop_at_epoch_end)
        assert metric_criteria.get_initial_value() in metric_threshold_to_subset_size_mapping
        self._metric_threshold_to_subset_size_mapping = metric_threshold_to_subset_size_mapping
        self._metric_criteria = metric_criteria

    def _get_subset_size_for_val_metric(self, val_metric: float) -> int:
        for threshold in sort_from_best_to_worst(
                list(self._metric_threshold_to_subset_size_mapping.keys()),
                self._metric_criteria):
            if val_metric >= threshold:
                return self._metric_threshold_to_subset_size_mapping[threshold]
        raise RuntimeError(
            f'Subset size not found but it is guaranteed to be available for '
            f'{self._metric_criteria.get_initial_value()}.')

    def get_next_iter(self, val_metric: t.Optional[float] = None) -> SizedIter[t.Tuple[Index, _T]]:
        val_metric = val_metric if val_metric is not None else self._metric_criteria.get_initial_value()
        return self._get_next_iter(self._get_subset_size_for_val_metric(val_metric))


class MetricBasedSubsetIterPlannerBuilder(IterPlannerBuilder):

    def __init__(
            self,
            metric_threshold_to_subset_size_mapping: t.Mapping[float, SubsetSize],
            metric_criteria: MetricCriteria,
            stop_at_epoch_end: bool = True) -> None:
        super().__init__()
        self._metric_threshold_to_subset_size_mapping = metric_threshold_to_subset_size_mapping
        self._metric_criteria = metric_criteria
        self._stop_at_epoch_end = stop_at_epoch_end

    def build(self, data_loader: DataLoader[_T]) -> IterPlanner[_T]:
        return MetricBasedSubsetIterPlanner(
            data_loader=data_loader,
            metric_threshold_to_subset_size_mapping={
                t: ss.as_nat(len(data_loader)) for t, ss in self._metric_threshold_to_subset_size_mapping.items()
            },
            metric_criteria=self._metric_criteria,
            stop_at_epoch_end=self._stop_at_epoch_end)
