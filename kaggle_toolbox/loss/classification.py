import typing as t

import torch

from .base import Loss


class BCEWithLogitsLoss(Loss):

    def __init__(
            self,
            weight: t.Optional[torch.Tensor] = None,
            pos_weight: t.Optional[torch.Tensor] = None) -> None:
        self._inner = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self._inner(y_pred, y_true)
