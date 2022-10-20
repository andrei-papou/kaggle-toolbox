import torch
import torch.nn.functional as torch_f

from .base import Loss


class MSELoss(Loss):

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch_f.mse_loss(y_pred, y_true, reduction='mean')


_SMOOTH_L1_BETA_DEFAULT = 1.0


class SmoothL1Loss(Loss):

    def __init__(self, beta: float = _SMOOTH_L1_BETA_DEFAULT):
        self._beta = beta

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch_f.smooth_l1_loss(y_pred, y_true, beta=self._beta, reduction='mean')
