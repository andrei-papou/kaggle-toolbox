from __future__ import annotations

import typing as t

import torch
import typing_extensions as t_ext
from torch.utils.data import default_collate as default_collate_fn

from kaggle_toolbox.data import Movable
from .tokenization.tokenizer import TokenizerResult


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


_TensorDict = t.Dict[str, torch.Tensor]


class TokenizerResultCollator:

    def __init__(self, base_collator: t.Callable[[t.List[_TensorDict]], _TensorDict] = default_collate_fn):
        self._base_collator = base_collator

    def __call__(self, item_list: t.List[TokenizerResult]) -> TokenizerResult:
        assert len(item_list) > 0
        tok_result_type = item_list[0].__class__
        return tok_result_type.from_collateable_dict(
            self._base_collator([item.to_collatable_dict() for item in item_list]))
