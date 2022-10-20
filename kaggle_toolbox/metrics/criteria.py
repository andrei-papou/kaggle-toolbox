import functools
import typing as t


class MetricCriteria:

    def is_improvement(self, old_val: float, new_val: float) -> bool:
        raise NotImplementedError()

    def get_initial_value(self) -> float:
        raise NotImplementedError()


class SmallerIsBetterCriteria(MetricCriteria):

    def is_improvement(self, old_val: float, new_val: float) -> bool:
        return old_val > new_val

    def get_initial_value(self) -> float:
        return float('inf')


class LargerIsBetterCriteria(MetricCriteria):

    def is_improvement(self, old_val: float, new_val: float) -> bool:
        return old_val < new_val

    def get_initial_value(self) -> float:
        return 0.0


def sort_from_best_to_worst(val_list: t.List[float], metric_criteria: MetricCriteria) -> t.List[float]:
    return sorted(
        val_list,
        key=functools.cmp_to_key(lambda l, r: 1 if metric_criteria.is_improvement(l, r) else -1))
