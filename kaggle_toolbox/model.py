import typing as t

import torch

_X = t.TypeVar('_X', contravariant=True)


class Model(torch.nn.Module, t.Generic[_X]):

    def forward(self, x: _X) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, x: _X) -> torch.Tensor:
        return super().__call__(x)
