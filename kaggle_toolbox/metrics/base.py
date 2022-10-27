import torch
import torchmetrics
import typing_extensions as t_ext

from .criteria import MetricCriteria


class Metric:

    def __enter__(self) -> t_ext.Self:
        return self

    def __exit__(self, *args, **kwargs):
        self.reset()

    def compute(self) -> float:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class MeanMetric(Metric):

    def __init__(self):
        self._inner = torchmetrics.MeanMetric()

    def __call__(self, x: torch.Tensor) -> None:
        return self._inner(x)

    def compute(self) -> float:
        return float(self._inner.compute().item())

    def reset(self):
        self._inner.reset()


class PredQualityMetric(Metric):
    name: str
    criteria: MetricCriteria

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        raise NotImplementedError()

    @classmethod
    def valid_name(cls) -> str:
        return f'valid_{cls.name}'


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
