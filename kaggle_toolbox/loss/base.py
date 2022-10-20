import torch


class Loss:

    def __call__(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
