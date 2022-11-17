from __future__ import annotations

import typing as t

import torch
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig

from .data import TokenizedX


_X = t.TypeVar('_X', bound=TokenizedX)


class Backbone(torch.nn.Module, t.Generic[_X]):

    def __init__(
            self,
            inner: torch.nn.Module,
            out_dim_size: int,
            initializer_range: t.Optional[float] = None):
        super().__init__()
        self._inner = inner
        self._out_dim_size = out_dim_size
        self._initializer_range = initializer_range

    @classmethod
    def from_huggingface_checkpoint(
            cls,
            checkpoint: str,
            zero_out_dropout: bool = False,
            output_hidden_states: bool = True) -> Backbone:
        config = AutoConfig.from_pretrained(checkpoint)
        if zero_out_dropout:
            config.hidden_dropout = 0.
            config.hidden_dropout_prob = 0.
            config.attention_dropout = 0.
            config.attention_probs_dropout_prob = 0.
        config.return_dict = True
        config.output_hidden_states = output_hidden_states
        return cls(
            AutoModel.from_pretrained(checkpoint, config=config),
            out_dim_size=config.hidden_size,
            initializer_range=getattr(config, 'initializer_range', None))

    @property
    def inner(self) -> torch.nn.Module:
        return self._inner

    @property
    def out_dim_size(self) -> int:
        return self._out_dim_size

    @property
    def initializer_range(self) -> t.Optional[float]:
        return self._initializer_range

    def named_parameters(self) -> t.Iterator[t.Tuple[str, torch.nn.parameter.Parameter]]:
        return self._inner.named_parameters()

    def forward(self, x: _X) -> torch.Tensor:
        return torch.stack(list(self._inner(**x.tensor_dict).hidden_states), dim=1)

    def __call__(self, x: _X) -> torch.Tensor:
        return super().__call__(x)
