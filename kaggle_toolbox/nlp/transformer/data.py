from __future__ import annotations

import typing as t

import torch
import typing_extensions as t_ext

from kaggle_toolbox.data import Movable


class TokenizedX(Movable, t_ext.Protocol):

    @property
    def input_ids(self) -> torch.Tensor:
        ...

    @property
    def attention_mask(self) -> torch.Tensor:
        ...

    @property
    def tensor_dict(self) -> t.Dict[str, torch.Tensor]:
        ...
