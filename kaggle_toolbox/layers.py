import torch


class SqueezeDim(torch.nn.Module):

    def __init__(self, dim: int = -1):
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self._dim)
