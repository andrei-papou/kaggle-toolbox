import typing as t

import torch

from .base import PredQualityMetric
from .criteria import SmallerIsBetterCriteria


class MCRMSEMetric(PredQualityMetric):
    name = 'mcrmse'
    criteria = SmallerIsBetterCriteria()

    def __init__(self):
        self._sq_diff_sum: t.Optional[torch.Tensor] = None
        self._n = 0

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        n, *_ = y_pred.shape
        sq_diff = (y_pred - y_true).pow(2).sum(dim=0)
        if self._sq_diff_sum is not None:
            self._sq_diff_sum += sq_diff
        else:
            self._sq_diff_sum = sq_diff
        self._n += n

    def compute(self) -> float:
        assert self._sq_diff_sum is not None and self._n > 0
        rmse_by_target = torch.sqrt(self._sq_diff_sum / self._n)
        return float(rmse_by_target.mean())

    def reset(self):
        self._sq_diff_sum = None
        self._n = 0
