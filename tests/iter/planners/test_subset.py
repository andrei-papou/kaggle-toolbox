import math
import typing as t

import torch
from torch.utils.data import Dataset, DataLoader

from kaggle_toolbox.iter.index import Index
from kaggle_toolbox.iter.planners.base import IterPlanner
from kaggle_toolbox.iter.planners.subset import FixedSubsetIterPlannerBuilder, \
    EpochBasedSubsetIterPlannerBuilder, MetricBasedSubsetIterPlannerBuilder
from kaggle_toolbox.iter.subset_size import NatSubsetSize, FracSubsetSize
from kaggle_toolbox.metrics import LargerIsBetterCriteria

T1 = t.TypeVar('T1', bound=t.SupportsFloat)
T2 = t.TypeVar('T2', bound=t.SupportsFloat)


def pair_is_close(lhs: t.Tuple[T1, T2], rhs: t.Tuple[T1, T2]) -> bool:
    return math.isclose(lhs[0], rhs[0]) and math.isclose(lhs[1], rhs[1])


def is_empty(it: t.Iterator[t.Any]) -> bool:
    return not any(True for _ in it)


class IntDataset(Dataset[int]):

    def __init__(self, size: int):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> int:
        return idx


def _unwrap_item(x: t.Tuple[Index, torch.Tensor]) -> t.Tuple[Index, t.List[int]]:
    return x[0], x[1].tolist()


class TestFixedSubsetIterPlanner:

    def test_fixed_subset_iter_planner_nat_do_end_of_epoch_stop(self) -> None:
        builder = FixedSubsetIterPlannerBuilder(subset_size=NatSubsetSize(2))
        data_loader = DataLoader(IntDataset(9), batch_size=2, num_workers=2)
        planner: IterPlanner[torch.Tensor] = builder.build(data_loader=data_loader)
        assert planner.epoch == 0
        assert planner.step == 0
        it = planner.get_next_iter()
        assert len(it) == 2
        assert planner.epoch == 0
        assert planner.step == 1
        assert _unwrap_item(next(it)) == (Index(0, (0, 5)), [0, 1])
        assert _unwrap_item(next(it)) == (Index(0, (1, 5)), [2, 3])
        assert is_empty(it)
        assert planner.epoch == 0
        assert planner.step == 1
        it = planner.get_next_iter()
        assert planner.epoch == 1
        assert planner.step == 2
        assert len(it) == 3
        assert _unwrap_item(next(it)) == (Index(0, (2, 5)), [4, 5])
        assert _unwrap_item(next(it)) == (Index(0, (3, 5)), [6, 7])
        assert _unwrap_item(next(it)) == (Index(0, (4, 5)), [8])
        assert is_empty(it)
        it = planner.get_next_iter()
        assert len(it) == 2
        assert _unwrap_item(next(it)) == (Index(1, (0, 5)), [0, 1])
        assert _unwrap_item(next(it)) == (Index(1, (1, 5)), [2, 3])
        assert is_empty(it)

    def test_fixed_subset_iter_planner_frac_do_end_of_epoch_stop(self) -> None:
        data_loader = DataLoader(IntDataset(9), batch_size=2, num_workers=2)
        planner: IterPlanner[torch.Tensor] = FixedSubsetIterPlannerBuilder(
            subset_size=FracSubsetSize(0.5)).build(data_loader=data_loader)
        it = planner.get_next_iter()
        assert len(it) == 2
        assert planner.epoch == 0
        assert planner.step == 1
        assert _unwrap_item(next(it)) == (Index(0, (0, 5)), [0, 1])
        assert _unwrap_item(next(it)) == (Index(0, (1, 5)), [2, 3])
        assert is_empty(it)
        assert planner.epoch == 0
        assert planner.step == 1
        it = planner.get_next_iter()
        assert planner.epoch == 1
        assert planner.step == 2
        assert len(it) == 3
        assert _unwrap_item(next(it)) == (Index(0, (2, 5)), [4, 5])
        assert _unwrap_item(next(it)) == (Index(0, (3, 5)), [6, 7])
        assert _unwrap_item(next(it)) == (Index(0, (4, 5)), [8])
        assert is_empty(it)
        it = planner.get_next_iter()
        assert len(it) == 2
        assert _unwrap_item(next(it)) == (Index(1, (0, 5)), [0, 1])
        assert _unwrap_item(next(it)) == (Index(1, (1, 5)), [2, 3])
        assert is_empty(it)


class TestEpochBasedSubsetIterPlanner:

    def test_epoch_based_subset_iter_planner(self) -> None:
        builder = EpochBasedSubsetIterPlannerBuilder(
            epoch_to_subset_size_mapping={
                0: NatSubsetSize(2),
                1: NatSubsetSize(3),
            })
        planner = builder.build(data_loader=DataLoader(IntDataset(5), batch_size=1))
        it = planner.get_next_iter()
        assert len(it) == 2
        assert next(it) == (Index(0, (0, 5)), 0)
        assert next(it) == (Index(0, (1, 5)), 1)
        assert is_empty(it)
        it = planner.get_next_iter()
        assert len(it) == 3
        assert next(it) == (Index(0, (2, 5)), 2)
        assert next(it) == (Index(0, (3, 5)), 3)
        assert next(it) == (Index(0, (4, 5)), 4)
        assert is_empty(it)
        it = planner.get_next_iter()
        assert len(it) == 5
        assert next(it) == (Index(1, (0, 5)), 0)
        assert next(it) == (Index(1, (1, 5)), 1)
        assert next(it) == (Index(1, (2, 5)), 2)
        assert next(it) == (Index(1, (3, 5)), 3)
        assert next(it) == (Index(1, (4, 5)), 4)
        assert is_empty(it)

    def test_metric_based_subset_iter_planner(self) -> None:
        builder = MetricBasedSubsetIterPlannerBuilder(
            metric_criteria=LargerIsBetterCriteria(),
            metric_threshold_to_subset_size_mapping={
                0.0: NatSubsetSize(1),
                0.5: NatSubsetSize(3),
            })
        planner = builder.build(data_loader=DataLoader(IntDataset(5), batch_size=1))
        it = planner.get_next_iter()
        assert len(it) == 1
        assert next(it) == (Index(0, (0, 5)), 0)
        assert is_empty(it)
        it = planner.get_next_iter(0.5)
        assert len(it) == 4
        assert next(it) == (Index(0, (1, 5)), 1)
        assert next(it) == (Index(0, (2, 5)), 2)
        assert next(it) == (Index(0, (3, 5)), 3)
        assert next(it) == (Index(0, (4, 5)), 4)
        assert is_empty(it)
