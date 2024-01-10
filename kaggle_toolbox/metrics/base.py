import typing as t

import torch
import torchmetrics

from kaggle_toolbox.data import DatasetKind
from .criteria import MetricCriteria, SmallerIsBetterCriteria

_S = t.TypeVar('_S', bound='Metric')


def format_dk_metric_name(name: str, dataset_kind: DatasetKind) -> str:
    return f'{dataset_kind.value}_{name}'


class Metric:
    criteria: MetricCriteria

    def __enter__(self: _S) -> _S:
        return self

    def __exit__(self, *args, **kwargs):
        self.reset()

    def compute(self) -> float:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class LossMetric(Metric):
    criteria: MetricCriteria = SmallerIsBetterCriteria()

    def __init__(self):
        self._inner = torchmetrics.MeanMetric()

    def __call__(self, x: torch.Tensor) -> None:
        return self._inner(x)

    def compute(self) -> float:
        return float(self._inner.compute().item())

    def reset(self):
        self._inner.reset()


class PredQualityMetric(Metric):

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        raise NotImplementedError()


class PredQualityTorchmetricsMetric(PredQualityMetric):

    @property
    def inner(self) -> torchmetrics.Metric:
        raise NotImplementedError()

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self.inner(y_pred, y_true)

    def compute(self) -> float:
        return float(self.inner.compute())

    def reset(self):
        self.inner.reset()
